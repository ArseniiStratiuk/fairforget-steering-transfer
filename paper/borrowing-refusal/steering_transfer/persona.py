"""Response-based persona vectors for a loaded causal language model.

Method: https://github.com/safety-research/persona_vectors at b8e0f044.
This implementation keeps generated token IDs, captures raw block outputs,
and joins positive/negative samples by ID rather than CSV row order.
Layers use zero-based indices. Model loading and experiment choices belong
in the notebook. No judge calls or model downloads happen on import.
"""

from contextlib import contextmanager
import math

import torch


@torch.inference_mode()
def generate_responses(
    model, tokenizer, prompts, *, seed, max_new_tokens=1000, temperature=1.0
):
    """Generate one response per rendered prompt; return text and exact IDs.

    Pass repeated prompts to obtain repeated samples. Sampling is explicit:
    temperature > 0 samples with top_p=1 and top_k=0; zero is greedy.
    Keep batch composition and seed fixed when comparing interventions.
    """
    if not prompts or max_new_tokens < 1 or temperature < 0:
        raise ValueError("Need prompts, a positive token cap and temperature >= 0")
    if tokenizer.padding_side != "left":
        raise ValueError("Batched generation requires left padding")
    torch.manual_seed(seed)
    inputs = tokenizer(
        prompts, padding=True, add_special_tokens=False, return_tensors="pt"
    ).to(model.device)
    options = {
        "do_sample": temperature > 0,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
    }
    if temperature > 0:
        options.update(temperature=temperature, top_p=1.0, top_k=0)
    outputs = model.generate(**inputs, **options, logits_to_keep=1)
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    rows = []
    for index, output in enumerate(outputs):
        prompt_ids = inputs.input_ids[index][inputs.attention_mask[index].bool()]
        generated = output[inputs.input_ids.shape[1] :].tolist()
        stop = next(
            (i for i, token in enumerate(generated) if token in eos_ids), len(generated)
        )
        response_ids = generated[:stop]
        rows.append(
            {
                "prompt_ids": prompt_ids.tolist(),
                "response_ids": response_ids,
                "answer": tokenizer.decode(response_ids, skip_special_tokens=True),
                "stop": "eos" if stop < len(generated) else "cap",
            }
        )
    return rows


@torch.inference_mode()
def capture_responses(model, layers, samples, *, prefix_lengths=None):
    """Return [sample, layer, hidden] means and last-prompt activations on CPU.

    Teacher forcing on saved IDs captures actual response token positions,
    excluding the prompt, EOS and padding. Each response has equal weight,
    regardless of length. Hooks avoid retaining every layer's full tensor.
    Raw block outputs also avoid final-model-normalization ambiguity.
    Optional prefix_lengths averages only the stated first response tokens,
    using the same forward pass; full response averages are always retained.
    """
    if not samples or not layers:
        raise ValueError("Need samples and decoder blocks")
    if prefix_lengths is not None and len(prefix_lengths) != len(samples):
        raise ValueError("Need one prefix length per sample")
    means, lasts, prefixes = [], [], []
    for sample_index, sample in enumerate(samples):
        prompt, response = sample["prompt_ids"], sample["response_ids"]
        if not prompt or not response:
            raise ValueError("Cannot capture an empty prompt or response")
        ids = torch.tensor([prompt + response], device=model.device)
        response_mean, prompt_last, prefix_mean, handles = [], [], [], []
        prefix_length = (
            len(response) if prefix_lengths is None else prefix_lengths[sample_index]
        )
        if not 1 <= prefix_length <= len(response):
            raise ValueError("Prefix must be inside the saved response")

        def capture(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            response_mean.append(hidden[0, len(prompt) :].float().mean(0).cpu())
            prompt_last.append(hidden[0, len(prompt) - 1].float().cpu())
            prefix_mean.append(
                hidden[0, len(prompt) : len(prompt) + prefix_length]
                .float()
                .mean(0)
                .cpu()
            )

        try:
            for layer in layers:
                handles.append(layer.register_forward_hook(capture))
            model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        if len(response_mean) != len(layers):
            raise RuntimeError("Not all decoder blocks executed")
        means.append(torch.stack(response_mean))
        lasts.append(torch.stack(prompt_last))
        prefixes.append(torch.stack(prefix_mean))
    return {
        "response_mean": torch.stack(means),
        "prompt_last": torch.stack(lasts),
        "prefix_mean": torch.stack(prefixes),
    }


def effective_pairs(positive, negative, *, threshold=50):
    """Join by pair_id and keep pos >= 50, neg < 50, both coherence >= 50.

    Matches the reference code's inclusive positive boundary. Missing scores
    fail the filter, never become zero. Returns IDs and rejection counts.
    """
    if not 0 <= threshold <= 100:
        raise ValueError("Threshold must be between 0 and 100")
    tables = []
    for records in (positive, negative):
        table = {row["pair_id"]: row for row in records}
        if len(table) != len(records):
            raise ValueError("Duplicate pair_id")
        tables.append(table)
    pos, neg = tables
    if pos.keys() != neg.keys():
        raise ValueError("Positive and negative pair IDs must match")
    kept, counts = [], {"invalid_score": 0, "low_coherence": 0, "no_contrast": 0}
    for pair_id in pos:
        scores = [
            row.get(key)
            for row in (pos[pair_id], neg[pair_id])
            for key in ("trait", "coherence")
        ]
        if any(
            value is None or not math.isfinite(value) or not 0 <= value <= 100
            for value in scores
        ):
            counts["invalid_score"] += 1
        elif scores[1] < 50 or scores[3] < 50:
            counts["low_coherence"] += 1
        elif scores[0] < threshold or scores[2] >= 100 - threshold:
            counts["no_contrast"] += 1
        else:
            kept.append(pair_id)
    return kept, {"total": len(pos), "kept": len(kept), **counts}


def mean_difference(positive, negative):
    """Equal-weight positive mean minus negative mean, without normalization.

    Inputs are [sample, layer, hidden]. Filter samples before calling.
    Preserve this raw difference: coefficient 1 means adding one group shift.
    """
    if positive.ndim != 3 or positive.shape != negative.shape or not len(positive):
        raise ValueError("Need matched nonempty [sample, layer, hidden] tensors")
    if not torch.isfinite(positive).all() or not torch.isfinite(negative).all():
        raise ValueError("Activations contain non-finite values")
    return positive.float().mean(0) - negative.float().mean(0)


def numeric_score(top_logprobs):
    """Return probability-weighted 0..100 score and retained numeric mass.

    Accept {token: log_probability}. Reference judge requires mass >= 0.25.
    Missing numeric mass gives None, never an invented rating.
    """
    mass, weighted = 0.0, 0.0
    for token, logprob in top_logprobs.items():
        try:
            value = int(token)
        except ValueError:
            continue
        if 0 <= value <= 100:
            probability = math.exp(logprob)
            mass += probability
            weighted += value * probability
    return (weighted / mass if mass >= 0.25 else None), mass


@contextmanager
def steer(layer, vector, coefficient, *, positions="response"):
    """Add coefficient * raw vector at one decoder block's output.

    Response mode changes the final prefill position and each cached decoding
    position. It assumes standard cached autoregressive generation. All mode
    changes every position, including the prompt. No weights change.
    """
    if positions not in ("response", "all") or vector.ndim != 1:
        raise ValueError("Need a 1D vector and positions='response' or 'all'")

    def add(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[-1] != vector.numel():
            raise ValueError("Vector width does not match the block")
        delta = coefficient * vector.to(device=hidden.device, dtype=hidden.dtype)
        shifted = hidden.clone()
        if positions == "response":
            shifted[:, -1, :] += delta
        else:
            shifted += delta
        return (shifted, *output[1:]) if isinstance(output, tuple) else shifted

    handle = layer.register_forward_hook(add)
    try:
        yield
    finally:
        handle.remove()


@torch.inference_mode()
def classify_responses(model, tokenizer, records, instruction, labels):
    """Return grammar-constrained labels for one batch of saved answers.

    LM Format Enforcer constructs the token constraint from the supplied labels.
    The instruction defines label meanings; output validity is not accuracy.
    """
    import json
    import re

    from lmformatenforcer import RegexParser
    from lmformatenforcer.integrations.transformers import (
        build_transformers_prefix_allowed_tokens_fn,
    )

    if not records or not labels or len(set(labels)) != len(labels):
        raise ValueError("Need records and distinct labels")
    prompts = [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": instruction
                    + json.dumps(
                        {"question": row["question"], "answer": row["answer"]}
                    ),
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in records
    ]
    inputs = tokenizer(
        prompts, padding=True, add_special_tokens=False, return_tensors="pt"
    ).to(model.device)
    parser = RegexParser("|".join(re.escape(label) for label in labels))
    prefix = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)
    generated = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=12,
        prefix_allowed_tokens_fn=prefix,
        logits_to_keep=1,
    )
    results = tokenizer.batch_decode(
        generated[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
    )
    if any(result not in labels for result in results):
        raise RuntimeError(f"Invalid constrained labels: {results}")
    return results

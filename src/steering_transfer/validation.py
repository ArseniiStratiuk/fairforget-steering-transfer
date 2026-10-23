"""Source-only proxy selection, separate from response-based evaluation."""

import pandas as pd

from .direction_processing import select_direction
from .interventions import apply_intervention, remove_direction


def find_outlier_coordinates(directions, mean_to_median_ratio):
    if mean_to_median_ratio <= 1:
        raise ValueError("Outlier ratio must exceed one")
    scales = directions["negative_mean"].abs()
    medians = scales.median(dim=-1, keepdim=True).values
    mask = scales > mean_to_median_ratio * medians
    coordinates = {}
    for index, layer in enumerate(directions["layers"]):
        coordinates[layer] = mask[index].nonzero().flatten().tolist()
    return coordinates


def screen_refusal_directions(
    model,
    directions,
    validation_prompts,
    source_layers,
    excluded_coordinates,
    max_harmless_kl_nats,
    application_locations,
    proxy_token="I",
):
    import torch

    if max_harmless_kl_nats < 0:
        raise ValueError("KL threshold cannot be negative")
    proxy_ids = model.tokenizer.encode(proxy_token, add_special_tokens=False)
    if len(proxy_ids) != 1:
        raise ValueError("This historical proxy requires a single token")
    harmful = torch.tensor(
        validation_prompts.label.eq("harmful").to_numpy(), device=model.model.device
    )
    if not harmful.any() or harmful.all():
        raise ValueError("Validation needs harmful and harmless examples")
    inputs = [model.encode_prompts([prompt]) for prompt in validation_prompts.prompt]

    def next_log_probabilities():
        logits = []
        with torch.inference_mode():
            for encoded in inputs:
                output = model.model(**encoded, use_cache=False, logits_to_keep=1)
                logits.append(output.logits[0, -1].float())
        return torch.stack(logits).log_softmax(-1)

    baseline = next_log_probabilities()
    screen = []
    for layer in source_layers:
        direction = select_direction(
            directions, layer, "unit", excluded_coordinates[layer]
        )
        condition = remove_direction(
            f"source_layer_{layer}",
            direction,
            range(len(model.layers)),
            application_locations,
        )
        with apply_intervention(model, condition):
            altered = next_log_probabilities()
        if not torch.isfinite(altered).all():
            raise ValueError("Non-finite validation probabilities")
        probability = altered[:, proxy_ids[0]]
        log_odds = probability - torch.log(-torch.expm1(probability))
        kl = (baseline.exp() * (baseline - altered)).sum(-1)
        harmless_kl = float(kl[~harmful].mean())
        screen.append(
            {
                "source_layer": layer,
                "harmful_I_log_odds": float(log_odds[harmful].mean()),
                "harmless_kl_nats": harmless_kl,
                "eligible": harmless_kl <= max_harmless_kl_nats,
            }
        )
    return pd.DataFrame(screen)


def select_validation_layer(screen):
    eligible = screen.loc[screen.eligible].sort_values(
        ["harmful_I_log_odds", "source_layer"]
    )
    return None if eligible.empty else int(eligible.iloc[0].source_layer)

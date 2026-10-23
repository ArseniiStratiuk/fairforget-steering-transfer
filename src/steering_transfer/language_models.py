"""Native model rendering and explicit ownership of GPU checkpoints."""

import gc
from pathlib import Path

from .result_storage import content_hash, find_project_root


class LanguageModel:
    def __init__(self, model, tokenizer, specification):
        self.model = model
        self.tokenizer = tokenizer
        self.specification = specification
        self.layers = model.model.language_model.layers
        self.tokenizer.padding_side = "left"
        self.closed = False

    def render_prompt(self, prompt):
        if self.closed:
            raise RuntimeError("Model is closed")
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def encode_prompts(self, prompts):
        rendered = [self.render_prompt(prompt) for prompt in prompts]
        return self.tokenizer(
            rendered, padding=True, add_special_tokens=False, return_tensors="pt"
        ).to(self.model.device)

    def close(self):
        if self.closed:
            return
        self.layers = None
        self.model = None
        self.closed = True
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()


def load_language_model(
    model,
    revision,
    checkpoint,
    precision="bfloat16",
    device="cuda:0",
    attention_implementation="sdpa",
):
    import torch
    import transformers
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

    checkpoint = Path(checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = find_project_root() / checkpoint
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Verified checkpoint is required: {checkpoint}")
    if checkpoint.name != revision:
        raise ValueError("Checkpoint directory must be named with its pinned revision")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    if not tokenizer.chat_template:
        raise ValueError(
            "This loader requires a native instruction-model chat template"
        )
    loaded = Gemma3ForConditionalGeneration.from_pretrained(
        checkpoint,
        dtype=getattr(torch, precision),
        device_map={"": device},
        attn_implementation=attention_implementation,
        local_files_only=True,
    ).eval()
    specification = {
        "model": model,
        "revision": revision,
        "tokenizer_revision": revision,
        "template_hash": content_hash(tokenizer.chat_template),
        "precision": precision,
        "attention_implementation": attention_implementation,
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "layers": len(loaded.model.language_model.layers),
        "hidden_size": loaded.config.text_config.hidden_size,
    }
    return LanguageModel(loaded, tokenizer, specification)

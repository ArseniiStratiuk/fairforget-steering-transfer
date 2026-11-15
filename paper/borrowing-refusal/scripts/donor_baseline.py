"""E0: an unmodified donor answers the whole test split (both languages), same decoding.

Usage: donor_baseline.py <model>   writes lapa_main/e0_<model>/responses
"""
import json
import os
import sys

from steering_transfer.datasets import load_frozen_prompts
from steering_transfer.interventions import unchanged_model
from steering_transfer.language_models import load_language_model
from steering_transfer.records import GreedyDecoding
from steering_transfer.response_generation import generate_responses
from steering_transfer.result_storage import file_hash, write_json

import run_lapa_main as main_code

model_name = sys.argv[1]
worker, workers = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
configuration = main_code.configuration
specification = configuration["models"][model_name]
test = load_frozen_prompts(main_code.root).query('split == "test"').copy()
test = test.sort_values(["language", "label", "prompt_id"]).iloc[worker::workers].reset_index(drop=True)
output = main_code.run / "lapa_main" / f"e0_{model_name}" / f"worker-{worker}"
import torch
batch = int(os.environ.get("GENERATION_BATCH", 32)) if torch.cuda.mem_get_info()[0] > 40 * 2**30 else 8
with load_language_model(model_name, specification["revision"], specification["checkpoint"]) as model:
    answers = generate_responses(
        model, test, [unchanged_model()],
        GreedyDecoding(configuration["generation"]["max_new_tokens"]), batch, output / "responses",
    )
write_json(output / "generation_complete.json", {
    "answers": len(answers), "model": model_name, "generation_batch": batch,
    "responses_sha256": file_hash(output / "responses/responses.jsonl"),
})
print("DONE e0", model_name, len(answers), flush=True)

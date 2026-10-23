"""Prompt and teacher-forced response capture using the same block definition."""

from pathlib import Path

from .result_storage import content_hash, exclusive_run, write_json


def capture_activations(model, records, layers, token_aggregation, output_directory):
    with exclusive_run(output_directory):
        import torch

        layers = list(layers)
        if records.empty or records.prompt_id.duplicated().any():
            raise ValueError("Capture requires distinct nonempty prompt IDs")
        if not layers or len(set(layers)) != len(layers):
            raise ValueError("Capture layers must be nonempty and distinct")
        if min(layers) < 0 or max(layers) >= len(model.layers):
            raise ValueError("Capture layer outside model")
        directory = Path(output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        provenance = {
            "model": model.specification,
            "records_hash": content_hash(records.to_dict("records")),
            "activation_location": "block_output",
            "token_aggregation": token_aggregation,
        }
        settings_hash = content_hash({**provenance, "layers": layers})
        manifest_path = directory / "capture.json"
        tensor_path = directory / "activations.pt"
        if tensor_path.exists():
            import json

            if (
                not manifest_path.exists()
                or json.loads(manifest_path.read_text())["settings_hash"]
                != settings_hash
            ):
                raise ValueError("Saved capture settings differ")
            return torch.load(tensor_path, map_location="cpu", weights_only=True)
        captured_records = []
        for record in records.to_dict("records"):
            if token_aggregation == "last_prompt_token":
                encoded = model.encode_prompts([record["prompt"]])
                start, end = encoded.input_ids.shape[1] - 1, encoded.input_ids.shape[1]
            else:
                if record.get("model_revision") != model.specification["revision"]:
                    raise ValueError(
                        "Saved response token IDs require their originating model revision"
                    )
                prompt_ids, response_ids = record["prompt_ids"], record["response_ids"]
                if not prompt_ids or not response_ids:
                    raise ValueError("Capture requires nonempty saved token IDs")
                start = len(prompt_ids)
                if token_aggregation == "response_mean":
                    end = start + len(response_ids)
                elif token_aggregation == "response_prefix_mean":
                    prefix_length = record["response_prefix_length"]
                    if not 1 <= prefix_length <= len(response_ids):
                        raise ValueError("Invalid response prefix length")
                    end = start + prefix_length
                else:
                    raise ValueError("Unknown token aggregation")
                ids = torch.tensor(
                    [prompt_ids + response_ids], device=model.model.device
                )
                encoded = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            captured = {}
            handles = []

            def capture_layer(layer_index):
                def capture(module, args, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    captured[layer_index] = (
                        hidden[0, start:end].detach().float().mean(0).cpu()
                    )

                return capture

            try:
                for layer_index in layers:
                    handles.append(
                        model.layers[layer_index].register_forward_hook(
                            capture_layer(layer_index)
                        )
                    )
                with torch.inference_mode():
                    model.model(**encoded, use_cache=False, logits_to_keep=1)
            finally:
                for handle in handles:
                    handle.remove()
            if set(captured) != set(layers):
                raise RuntimeError("Not all requested capture layers executed")
            captured_records.append(torch.stack([captured[index] for index in layers]))
        values = torch.stack(captured_records)
        if not torch.isfinite(values).all():
            raise ValueError("Non-finite captured activations")
        bundle = {
            "activations": values,
            "record_ids": records.prompt_id.tolist(),
            "layers": layers,
            "provenance": provenance,
        }
        temporary = directory / "activations.tmp"
        torch.save(bundle, temporary)
        temporary.replace(tensor_path)
        write_json(
            manifest_path, {"settings_hash": settings_hash, "provenance": provenance}
        )
        return bundle


def capture_prompt_activations(
    model, prompts, layers, activation_location, token_position, output_directory
):
    if activation_location != "block_output" or token_position != "last_prompt_token":
        raise ValueError(
            "Implemented prompt capture is last_prompt_token at block_output"
        )
    return capture_activations(model, prompts, layers, token_position, output_directory)


def capture_response_activations(
    model, responses, layers, activation_location, token_aggregation, output_directory
):
    if activation_location != "block_output":
        raise ValueError("Only raw block-output response capture is implemented")
    if token_aggregation not in {"response_mean", "response_prefix_mean"}:
        raise ValueError("Response capture needs response_mean or response_prefix_mean")
    return capture_activations(
        model, responses, layers, token_aggregation, output_directory
    )

"""Contrastive means and explicitly recorded direction processing."""

from .records import Direction
from .result_storage import file_hash


def compute_contrastive_directions(activations, positive_ids, negative_ids):
    import torch

    positive_ids, negative_ids = list(positive_ids), list(negative_ids)
    if not positive_ids or not negative_ids:
        raise ValueError("Both contrast groups must be nonempty")
    if len(set(positive_ids)) != len(positive_ids) or len(set(negative_ids)) != len(
        negative_ids
    ):
        raise ValueError("Contrast groups contain repeated IDs")
    if set(positive_ids) & set(negative_ids):
        raise ValueError("Contrast groups overlap")
    indices = {
        record_id: index for index, record_id in enumerate(activations["record_ids"])
    }
    if len(indices) != len(activations["record_ids"]):
        raise ValueError("Activation IDs are not unique")
    positive = [indices[record_id] for record_id in positive_ids]
    negative = [indices[record_id] for record_id in negative_ids]
    values = activations["activations"].float()
    differences = values[positive].mean(0) - values[negative].mean(0)
    if not torch.isfinite(differences).all():
        raise ValueError("Non-finite contrastive direction")
    return {
        "directions": differences,
        "positive_mean": values[positive].mean(0),
        "negative_mean": values[negative].mean(0),
        "layers": activations["layers"],
        "provenance": {
            **activations["provenance"],
            "positive_ids": positive_ids,
            "negative_ids": negative_ids,
            "contrast": "positive_mean_minus_negative_mean",
        },
    }


def select_direction(directions, layer, normalization, excluded_coordinates):
    import torch

    index = directions["layers"].index(layer)
    values = directions["directions"][index].detach().float().clone()
    for coordinate in excluded_coordinates:
        if coordinate < 0 or coordinate >= values.numel():
            raise ValueError("Excluded coordinate outside hidden width")
        values[coordinate] = 0
    norm = float(values.norm())
    if not torch.isfinite(values).all() or norm == 0:
        raise ValueError("Direction must be finite and nonzero")
    if normalization == "unit":
        values /= norm
    elif normalization != "raw":
        raise ValueError("normalization must be unit or raw")
    return Direction(
        values,
        {
            **directions["provenance"],
            "layer": layer,
            "normalization": normalization,
            "raw_norm": norm,
            "excluded_coordinates": list(excluded_coordinates),
        },
    )


def create_random_directions(reference_direction, seeds, excluded_coordinates):
    import torch

    if len(set(seeds)) != len(seeds):
        raise ValueError("Random seeds must be distinct")
    random_directions = {}
    reference = reference_direction.values.detach().float().cpu()
    for seed in seeds:
        generator = torch.Generator().manual_seed(seed)
        values = torch.randn(reference.shape, generator=generator)
        values[list(excluded_coordinates)] = 0
        if values.norm() == 0:
            raise ValueError("Mask removed the entire random direction")
        values = values / values.norm() * reference.norm()
        random_directions[seed] = Direction(
            values,
            {
                "method": "gaussian_norm_matched",
                "seed": seed,
                "excluded_coordinates": list(excluded_coordinates),
                "reference_provenance": reference_direction.provenance,
            },
        )
    return random_directions


def load_direction_set(path):
    import torch

    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if not {"directions", "layers", "provenance"}.issubset(bundle):
        raise ValueError("Expected a versioned direction set, not an unnamed tensor")
    bundle["provenance"] = {**bundle["provenance"], "file_sha256": file_hash(path)}
    return bundle

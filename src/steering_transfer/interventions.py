"""Explicit direction removal and addition with scoped hooks."""

from contextlib import contextmanager

from .records import Condition

REMOVAL_LOCATIONS = (
    "block_input",
    "attention_residual_contribution",
    "feedforward_residual_contribution",
)


def unchanged_model(name="baseline"):
    return Condition(name=name)


def remove_direction(name, direction, layers, application_locations):
    locations = tuple(application_locations)
    if not locations or not set(locations).issubset(REMOVAL_LOCATIONS):
        raise ValueError("Unknown or empty removal locations")
    values = direction.values.float()
    if abs(float(values.norm()) - 1.0) > 1e-5:
        raise ValueError("Projection removal requires a unit direction")
    return Condition(name, "remove_direction", direction, tuple(layers), locations)


def add_direction(name, direction, layer, strength, token_positions):
    import math

    if token_positions not in {"response", "all"} or not math.isfinite(strength):
        raise ValueError("Invalid token positions or strength")
    return Condition(
        name,
        "add_direction",
        direction,
        (layer,),
        ("block_output",),
        strength,
        token_positions,
    )


@contextmanager
def apply_intervention(model, condition):
    import torch

    if model.closed:
        raise RuntimeError("Cannot intervene on a closed model")
    if condition.method == "unchanged":
        yield
        return
    if not condition.layers or len(set(condition.layers)) != len(condition.layers):
        raise ValueError("Intervention requires distinct layers")
    if min(condition.layers) < 0 or max(condition.layers) >= len(model.layers):
        raise ValueError("Layer outside model")
    vector = condition.direction.values
    if vector.ndim != 1 or not torch.isfinite(vector).all():
        raise ValueError("Direction must be a finite vector")
    if vector.numel() != model.specification["hidden_size"]:
        raise ValueError("Direction and target model hidden widths differ")
    handles = []
    calls = 0
    device_vector = vector.to(model.model.device, dtype=torch.float32)

    def project(hidden):
        nonlocal calls
        direction = device_vector.to(device=hidden.device)
        values = hidden.float()
        calls += 1
        projection = (values @ direction).unsqueeze(-1) * direction
        return (values - projection).to(hidden.dtype)

    def project_input(module, args, kwargs):
        if args:
            return (project(args[0]), *args[1:]), kwargs
        return args, {**kwargs, "hidden_states": project(kwargs["hidden_states"])}

    def project_output(module, args, output):
        return project(output)

    def add_output(module, args, output):
        nonlocal calls
        hidden = output[0] if isinstance(output, tuple) else output
        delta = condition.strength * device_vector.to(hidden.device, hidden.dtype)
        shifted = hidden.clone()
        if condition.token_positions == "response":
            shifted[:, -1, :] += delta
        else:
            shifted += delta
        calls += 1
        if isinstance(output, tuple):
            return shifted, *output[1:]
        return shifted

    try:
        for layer_index in condition.layers:
            layer = model.layers[layer_index]
            if condition.method == "add_direction":
                handles.append(layer.register_forward_hook(add_output))
            elif condition.method == "remove_direction":
                if "block_input" in condition.application_locations:
                    handles.append(
                        layer.register_forward_pre_hook(project_input, with_kwargs=True)
                    )
                if "attention_residual_contribution" in condition.application_locations:
                    handles.append(
                        layer.post_attention_layernorm.register_forward_hook(
                            project_output
                        )
                    )
                if (
                    "feedforward_residual_contribution"
                    in condition.application_locations
                ):
                    handles.append(
                        layer.post_feedforward_layernorm.register_forward_hook(
                            project_output
                        )
                    )
            else:
                raise ValueError(f"Unsupported intervention: {condition.method}")
        yield
        if not calls:
            raise RuntimeError("Intervention was installed but never executed")
    finally:
        for handle in handles:
            handle.remove()

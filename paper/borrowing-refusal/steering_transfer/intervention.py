from contextlib import contextmanager


@contextmanager
def ablate(layers, direction):
    """Project a unit direction out of every block input and residual contribution.

    Activations change only inside this context. Weights remain unchanged and
    hooks are removed even if inference fails. ``calls`` counts each hook site.
    """
    handles = []
    calls = {"block_input": 0, "attention": 0, "mlp": 0}

    def project(hidden):
        values = hidden.float()
        return (values - (values @ direction)[..., None] * direction).to(hidden.dtype)

    def block_input(module, args, kwargs):
        calls["block_input"] += 1
        if args:
            return (project(args[0]), *args[1:]), kwargs
        return args, {**kwargs, "hidden_states": project(kwargs["hidden_states"])}

    def attention_output(module, args, output):
        calls["attention"] += 1
        return project(output)

    def mlp_output(module, args, output):
        calls["mlp"] += 1
        return project(output)

    try:
        for layer in layers:
            handles.append(
                layer.register_forward_pre_hook(block_input, with_kwargs=True)
            )
            handles.append(
                layer.post_attention_layernorm.register_forward_hook(attention_output)
            )
            handles.append(
                layer.post_feedforward_layernorm.register_forward_hook(mlp_output)
            )
        yield calls
    finally:
        for handle in handles:
            handle.remove()

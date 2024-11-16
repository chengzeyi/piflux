import torch
from torch.overrides import TorchFunctionMode

piflux_ops = torch.ops.piflux


class DistributedAttentionMode(TorchFunctionMode):
    @torch.compiler.disable()
    def __init__(self):
        super().__init__()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        if func is torch.nn.functional.scaled_dot_product_attention:
            (query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa) = get_args(
                args, kwargs, "query", "key", "value", "attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa"
            )

            assert (
                attn_mask is None
            ), "attn_mask is not supported in DistributedAttentionMode for scaled_dot_product_attention"
            assert (
                is_causal is None or is_causal is False
            ), "is_causal is not supported in DistributedAttentionMode for scaled_dot_product_attention"

            key = piflux_ops.get_complete_tensor(key, dim=-2, enable_cache=True)
            value = piflux_ops.get_complete_tensor(value, dim=-2, enable_cache=True)

            args = (query, key, value)
            kwargs = {}
            if dropout_p is not None:
                kwargs["dropout_p"] = dropout_p
            if is_causal is not None:
                kwargs["is_causal"] = is_causal
            if scale is not None:
                kwargs["scale"] = scale
            if enable_gqa is not None:
                kwargs["enable_gqa"] = enable_gqa
            return func(*args, **kwargs)

        return func(*args, **kwargs)

    @torch.compiler.disable()
    def __enter__(self):
        super().__enter__()

    @torch.compiler.disable()
    def __exit__(self, *args):
        super().__exit__(*args)


def get_arg(args, kwargs, *field):
    if len(field) == 1:
        if isinstance(field, int):
            if field < len(args):
                return args[field]
            else:
                return None
        else:
            return kwargs.get(field[0])
    else:
        index, name = field
        if index < len(args):
            return args[index]
        else:
            return kwargs.get(name)


def get_args(args, kwargs, *names):
    results = []
    for i, name in enumerate(names):
        results.append(get_arg(args, kwargs, i, name))
    return results

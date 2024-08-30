import torch
import torch.distributed as dist
from torch.overrides import TorchFunctionMode

from . import context


class DistributedAttentionMode(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        ctx = context.current_context

        if func is torch.nn.functional.scaled_dot_product_attention and ctx.world_size > 1:
            idx = ctx.counters["scaled_dot_product_attention"]
            ctx.counters["scaled_dot_product_attention"] += 1

            (query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa) = get_args(
                args, kwargs, "query", "key", "value", "attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa"
            )

            assert attn_mask is None, "attn_mask is not supported in distributed mode for scaled_dot_product_attention"

            key = key.contiguous()
            value = value.contiguous()

            gathered_keys = context.get_buffer_list(f"scaled_dot_product_attention_keys_{idx}", key)
            gathered_values = context.get_buffer_list(f"scaled_dot_product_attention_values_{idx}", value)

            dist.all_gather(gathered_keys, key)
            dist.all_gather(gathered_values, value)

            key = torch.cat(gathered_keys, dim=2)
            value = torch.cat(gathered_values, dim=2)

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


def get_arg(args, kwargs, *field):
    if field == 1:
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

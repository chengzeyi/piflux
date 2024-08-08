from typing import Type

import torch


def make_parallel_flux_transformer_block(module_cls: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    class ParallelFluxTransformerBlock(module_cls):
        _parent = module_cls

        def forward(
            self,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor,
            temb: torch.FloatTensor,
            image_rotary_emb=None,
        ):
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, emb=temb
            )

            # Attention.
            attn_output, context_attn_output = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
            )

            # Process attention outputs for the `hidden_states`.
            attn_output = gate_msa.unsqueeze(1) * attn_output
            hidden_states = hidden_states + attn_output

            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            ff_output = self.ff(norm_hidden_states)
            ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = hidden_states + ff_output

            # Process attention outputs for the `encoder_hidden_states`.

            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

            context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

            return encoder_hidden_states, hidden_states

    return ParallelFluxTransformerBlock

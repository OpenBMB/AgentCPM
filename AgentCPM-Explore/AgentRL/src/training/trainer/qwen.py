from typing import Optional
from torch.distributed.tensor.parallel import ColwiseParallel,RowwiseParallel,SequenceParallel,PrepareModuleInput,PrepareModuleOutput,PrepareModuleInputOutput
from torch.distributed.tensor import Replicate,Shard,DTensor
import torch.distributed.fsdp
from torch.distributed.fsdp import MixedPrecisionPolicy

from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM,Qwen3Attention,Qwen3MLP
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM,Qwen2Attention,Qwen2MLP

from training.arl import AsyncTrainer,logger
from training.parallel.context import AttentionContextParallel
from collections import defaultdict

class QwenTrainer(AsyncTrainer):
    @property
    def layer_tp_plan(self):
        atte_dict = defaultdict(lambda: None)
        atte_dict["hidden_states"] = Shard(1)
        if self.args.attention_implementation != "sdpa":
            atte_dict["attention_mask"] = Replicate()
        atte_desired_dict = defaultdict(lambda: None)
        atte_desired_dict["hidden_states"] = Replicate()
        if self.args.attention_implementation != "sdpa":
            atte_desired_dict["attention_mask"] = Replicate()
        logger.warning("Tensor Parallel Enabled, forward parameter `logits_to_keep` may not be processed correctly.")

        layer_tp_plan = {
            # "model.norm": SequenceParallel(),
            # "lm_head": ColwiseParallel(
            #     input_layouts=Shard(1),
            #     use_local_output=False
            # ),
            "model.norm": SequenceParallel(
                use_local_output=True
            ),
            "lm_head": ColwiseParallel(
                input_layouts=Shard(1),
                use_local_output=False
            ),
            "model.embed_tokens": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "model.rotary_emb": PrepareModuleOutput(
                output_layouts=(Replicate(),Replicate()),
                desired_output_layouts=(Replicate(),Replicate()),
                use_local_output=False
            ),
            "model.layers.*.mlp": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "model.layers.*.mlp.up_proj": ColwiseParallel(),
            "model.layers.*.mlp.gate_proj": ColwiseParallel(),
            "model.layers.*.mlp.down_proj": RowwiseParallel(
                output_layouts=Shard(1),
            ),
            "model.layers.*.self_attn": PrepareModuleInput(
                input_kwarg_layouts=atte_dict,
                desired_input_kwarg_layouts=atte_desired_dict,
            ),
            "model.layers.*.self_attn.q_proj": ColwiseParallel(use_local_output=False),
            "model.layers.*.self_attn.k_proj": ColwiseParallel(use_local_output=False),
            "model.layers.*.self_attn.v_proj": ColwiseParallel(use_local_output=False),
            "model.layers.*.self_attn.q_norm": SequenceParallel(sequence_dim=2),
            "model.layers.*.self_attn.k_norm": SequenceParallel(sequence_dim=2),
            "model.layers.*.self_attn.o_proj": RowwiseParallel(
                output_layouts=Shard(1),
            ),
            "model.layers.*.input_layernorm": SequenceParallel(),
            "model.layers.*.post_attention_layernorm": SequenceParallel(),
        }
        return layer_tp_plan

    @property
    def layer_cp_plan(self):
        layer_cp_plan = {
            "model.layers.*.self_attn": AttentionContextParallel(),
        }
        # we need to shard sequence in the sequnce dim
        layer_cp_plan.update({
            "model.embed_tokens": PrepareModuleInput(
                input_layouts=Replicate(),
                desired_input_layouts=Shard(1),
                use_local_output=True
            ),
            "model.rotary_emb": PrepareModuleOutput(
                output_layouts=(Replicate(),Replicate()),
                desired_output_layouts=(Shard(1),Shard(1)),
                use_local_output=True
            )
        })
        return layer_cp_plan

    def chunk_model(self, model: Qwen3ForCausalLM|Qwen2ForCausalLM):
        """Chunk the model into pipeline stages for pipeline parallelism."""
        class QwenSubModule(torch.nn.Module):
            def __init__(
                self,
                module:Qwen3ForCausalLM|Qwen2ForCausalLM,
                mesh: torch.distributed.device_mesh.DeviceMesh,
                start_idx:int,
                end_idx:int,
                is_first:bool=False,
                is_last:bool=False
            ):
                super().__init__()
                self.module = module
                self.mesh = mesh
                self.start_idx = start_idx
                self.end_idx = end_idx
                self.is_first = is_first
                self.is_last = is_last

                if not is_first:
                    self.module.model.embed_tokens = None
                    logger.debug("Removed embedding layer for non-first stage.")

                if not is_last:
                    self.module.lm_head = None
                    self.module.model.norm = None
                    logger.debug("Removed lm_head layer and norm layer for non-last stage.")

                self.module.model.layers = torch.nn.ModuleList(
                    [None]*start_idx + [*self.module.model.layers[start_idx:end_idx]] + [None] * (len(self.module.model.layers) - end_idx)
                )
                logger.debug(f"Pipeline stage layers from {start_idx} to {end_idx} initialized.")

            def forward(
                self,
                hidden_states: Optional[torch.FloatTensor] = None,
                position_embeddings: Optional[torch.FloatTensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                input_ids: Optional[torch.LongTensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                **kwargs,
            ):
                if self.is_first:
                    # Forward pass for the first stage
                    assert input_ids is not None or inputs_embeds is not None, "Input IDs or input embeddings must be provided for the first stage."
                    if inputs_embeds is None:
                        inputs_embeds = self.module.model.embed_tokens(input_ids)

                    if position_embeddings is None:
                        if position_ids is None:
                            position_ids = torch.arange(0,inputs_embeds.size(1),device=inputs_embeds.device).unsqueeze(0).expand(inputs_embeds.size(0),-1)
                        position_embeddings = self.module.model.rotary_emb(inputs_embeds, position_ids)
                        if isinstance(position_embeddings, tuple):
                            position_embeddings = torch.stack(position_embeddings, dim=0)

                    hidden_states = inputs_embeds
                else:
                    assert hidden_states is not None, "Hidden states must be provided for non-first stages."
                    assert position_embeddings is not None, "Position embeddings must be provided for non-first stages."

                    if self.mesh["tp"].size() > 1 and not isinstance(hidden_states, DTensor):
                        hidden_states:DTensor = DTensor.from_local(hidden_states, self.mesh["tp"])
                        hidden_states = hidden_states.redistribute(placements=[Shard(1)])

                pe_for_layer = position_embeddings if position_embeddings.dim() <= 3 else (position_embeddings[0], position_embeddings[1])
                for layer in self.module.model.layers[self.start_idx:self.end_idx]:
                    hidden_states = layer(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        position_embeddings=pe_for_layer,
                        use_cache=False
                    )
                    if isinstance(hidden_states, tuple):
                        hidden_states = hidden_states[0]

                if self.is_last:
                    hidden_states = self.module.model.norm(hidden_states)
                    logits = self.module.lm_head(hidden_states)
                    return logits
                else:
                    hidden_states = hidden_states.full_tensor() if isinstance(hidden_states, DTensor) else hidden_states
                    return hidden_states, position_embeddings



        # calculate how many layers should be in each chunk
        # embedding layer, output layer stand for two layers in this computation

        hidden_layers = len(model.model.layers)
        embed_virtual = 2          # Number of virtual layers for embedding
        tail_virtual = 2           # Number of virtual layers for norm + lm_head
        virtual_total = hidden_layers + embed_virtual + tail_virtual

        base = virtual_total // self.args.pp_size
        rem = virtual_total % self.args.pp_size  # First rem stages allocated (base+1) virtual layers

        r = self.pp_rank
        if r < rem:
            v_len = base + 1
            v_start = r * (base + 1)
        else:
            v_len = base
            v_start = rem * (base + 1) + (r - rem) * base
        v_end = v_start + v_len  # Half-open interval

        # Virtual layer interval division:
        # [0, embed_virtual) ------------------> embedding virtual layers
        # [embed_virtual, embed_virtual + hidden_layers) -> real transformer layers
        # [embed_virtual + hidden_layers, virtual_total) -> tail (norm + lm_head) virtual layers
        transformer_region_start = embed_virtual
        transformer_region_end = embed_virtual + hidden_layers

        # Calculate the real transformer layer range covered by this stage (half-open interval)
        t_start_virtual = max(v_start, transformer_region_start)
        t_end_virtual = min(v_end, transformer_region_end)

        if t_start_virtual < t_end_virtual:
            start_idx = t_start_virtual - transformer_region_start
            end_idx = t_end_virtual - transformer_region_start
        else:
            # This stage does not contain any transformer layers
            start_idx = end_idx = 0
            logger.warning(f"Pipeline rank {r} not allocated any real transformer layers (virtual layer interval: [{v_start},{v_end}))")

        return QwenSubModule(
            model,
            mesh=self.mesh,
            start_idx=start_idx,
            end_idx=end_idx,
            is_first=self.pp_rank == 0,
            is_last=self.pp_rank == self.args.pp_size - 1
        )

    def check_fn(self, module):
        return isinstance(module,(
            Qwen3Attention,Qwen3MLP,
            Qwen2Attention,Qwen2MLP,
        ))

    def prepare_fsdp(self, model: Qwen3ForCausalLM|Qwen2ForCausalLM):
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        reshard_after_forward = self.accelerator.state.fsdp_plugin.reshard_after_forward if self.accelerator.state.fsdp_plugin is not None else True
        for layer in model.model.layers:
            if layer is not None:
                torch.distributed.fsdp.fully_shard(
                    layer,
                    mesh=self.mesh["dp"],
                    mp_policy=mp_policy,
                    reshard_after_forward=reshard_after_forward
                )

        if model.model.embed_tokens is not None:
            torch.distributed.fsdp.fully_shard(
                model.model.embed_tokens,
                mesh=self.mesh["dp"],
                mp_policy=mp_policy,
                reshard_after_forward=reshard_after_forward
            )

        if model.lm_head is not None:
            torch.distributed.fsdp.fully_shard(
                model.lm_head,
                mesh=self.mesh["dp"],
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.float32,
                    reduce_dtype=torch.float32,
                ),
                reshard_after_forward=reshard_after_forward
            )
        if self.args.pp_size == 1:
            torch.distributed.fsdp.fully_shard(
                model,
                mesh=self.mesh["dp"],
                mp_policy=mp_policy,
                reshard_after_forward=reshard_after_forward
            )
        return model

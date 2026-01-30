import os
import torch
import transformers
import contextlib
import asyncio
import time
import shutil
import tqdm
import inspect
import warnings
import gc

warnings.simplefilter("once", UserWarning)


from abc import abstractmethod, ABC
from copy import deepcopy
from fnmatch import fnmatch

from typing import Callable, Optional, Union, Type, Literal, Iterable, AsyncGenerator, Any, Coroutine
from collections import defaultdict
from collections.abc import Iterable
from packaging import version
from functools import partial
import torch.distributed as dist
from torch.distributed.tensor import DTensor, distribute_tensor, Replicate, Shard
from torch.distributed.pipelining import PipelineStage
from torch.utils.data import Dataset, IterableDataset
from torch.distributed.tensor.parallel import parallelize_module, loss_parallel, ParallelStyle
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    Trainer
)
from torch.nn.attention import sdpa_kernel, SDPBackend
import torch.distributed.nn.functional

from transformers.trainer import (
    WEIGHTS_NAME,
    TRAINING_ARGS_NAME,
    SAFE_WEIGHTS_NAME,
    is_peft_available,
)

if is_peft_available():
    from peft import PeftModel


from transformers.trainer import seed_worker, is_datasets_available, DataLoader, speed_metrics

from accelerate.utils.memory import clear_device_cache
from configs import AgentTrainingConfig
from databases import Task, Record, DistributedLock, DistributedCounter, DataLabels, init_data_models, init_databases
from log import logger, set_process_title
from .utils import  efficient_loading, auto_broadcast
from .datasets import DBIterableDataset
from .inference import InferenceManager
from .parallel.pipe import IterSchedule1F1B

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]
torch.backends.cuda.enable_flash_sdp(True)


class AsyncTrainer(Trainer,ABC):
    def __init__(
        self,
        model: PreTrainedModel,
        args: AgentTrainingConfig = None,
        full_state_dict: Optional[dict] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        convert_record_to_data_func: Optional[AsyncGenerator[dict[str, Any], Any]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[list[TrainerCallback]] = [],
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        loop: Optional[asyncio.AbstractEventLoop] = None,
        **kwargs,
    ):
        """
        Initialize the AsyncGRPOTrainer for agent reinforcement learning.
        Args:
            model (PreTrainedModel): The pre-trained model to be trained.
            sampler_class (Type[AsyncSampler]): Class for asynchronous sampling during training.
            calculate_score (Callable[[Record], asyncio.Future[float]]): Function to calculate reward scores for records.
            args (AgentTrainingConfig, optional): Training configuration parameters. Defaults to None.
            train_dataset (Optional[Union[Dataset, IterableDataset, dict]], optional): Training dataset. Defaults to None.
            eval_dataset (Optional[Union[Dataset, IterableDataset, dict]], optional): Evaluation dataset. Defaults to None.
            convert_record_to_data_func (Optional[AsyncGenerator[dict[str, Any], Any]], optional): Function to convert records to training data. Defaults to None. See `src/training/datasets.py` for examples.
            processing_class (Optional[PreTrainedTokenizerBase], optional): Tokenizer or processing class. Defaults to None.
            callbacks (Optional[list[TrainerCallback]], optional): List of training callbacks. Defaults to None.
            optimizers (tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]], optional): Optimizer and scheduler tuple. Defaults to (None, None).
            data_collator (Optional[Callable[[list], dict]], optional): Function to collate data batches. Defaults to None.
            loop (Optional[asyncio.AbstractEventLoop], optional): Event loop for async operations. Defaults to None.
            **kwargs: Additional keyword arguments passed to the parent class.
        Raises:
            ValueError: If an unsupported loss_calculater is specified in args.
        Notes:
            - Initializes distributed training with tensor parallelism and data parallelism
            - Sets up reward calculation process in a separate worker
            - Configures model parallelization based on tp_size
            - Creates device meshes for distributed communication
            - Supports GRPO and CrossEntropy loss calculation methods
            - Enables sampling and inference management when specified in config
        """
        set_process_title(args)
        self.model_cfg = model.config
        self.model_class = model.__class__
        self.torch_dtype = model.dtype
        self.global_state_dict = model.state_dict()

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self.loop = loop if loop else asyncio.new_event_loop()
        self.args = args if args else AgentTrainingConfig()
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK",))
        self.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE",))
        self.full_state_dict = full_state_dict
        
        self.loop.run_until_complete(init_databases(args))
        if self.rank == 0:
            self.loop.run_until_complete(init_data_models(clean_all=args.resume_from_checkpoint is None and args.enable_sampling))
        dist.barrier()

        # create device meshes
        self.mesh = dist.device_mesh.init_device_mesh(
            device_type="cuda",
            mesh_shape=(
                self.world_size // (self.args.tp_size * self.args.pp_size * self.args.ep_size * self.args.cp_size),
                self.args.ep_size,
                self.args.pp_size,
                self.args.cp_size,
                self.args.tp_size,
            ),
            mesh_dim_names=["dp", "ep", "pp", "cp", "tp",] 
        )
        self.tp_group = self.mesh.get_group("tp")
        self.cp_group = self.mesh.get_group("cp")
        self.pp_group = self.mesh.get_group("pp")
        self.ep_group = self.mesh.get_group("ep")
        self.tp_rank = self.mesh.get_local_rank("tp")
        self.cp_rank = self.mesh.get_local_rank("cp")
        self.pp_rank = self.mesh.get_local_rank("pp")
        self.ep_rank = self.mesh.get_local_rank("ep")
        self.dp_rank = self.mesh.get_local_rank("dp")
        self.is_dp_master = self.tp_rank == 0 and self.cp_rank == 0 and self.pp_rank == 0 and self.ep_rank == 0
        
        # if args.max_grad_norm is not None and args.max_grad_norm > 0:
        import patches.norm
        torch.nn.utils.clip_grad_norm_ =  partial(patches.norm.clip_grad_norm_,pp_pg=self.pp_group)
        
        self.origin_train_dataset = train_dataset
        db_trainset = DBIterableDataset(
            args=self.args,
            split="train",
            dp_rank=self.dp_rank,
            dp_size=self.mesh["dp"].size(),
            fetching=self.is_dp_master,
            convert_record_to_data_func=convert_record_to_data_func,
            processing_class=processing_class,
        )
        
        if args.eval_strategy != "no":
            class DummyEvalDataset(IterableDataset):
                def __iter__(self):
                    while True:
                        yield {}
            eval_dataset = DummyEvalDataset()
        
        super().__init__(
            model=model,
            args=args,
            train_dataset=db_trainset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            compute_loss_func=self.unified_loss_calculation,
            **kwargs,
        )
        
        self.accelerator.gradient_accumulation_steps = args.gradient_accumulation_steps
        dist.barrier()
        if args.enable_sampling or (eval_dataset is not None and self.args.eval_strategy != "no"):
            self.inf_manager = InferenceManager(
                config=self.args,
                loop=self.loop
            )
        self.sampling_sem = None
        dist.barrier()
        
    @property
    def layer_tp_plan(self) -> dict[str, ParallelStyle]:
        raise NotImplementedError("Subclasses must implement the layer_tp_plan property to define the tensor parallelism plan for the model layers.")

    @property
    def layer_cp_plan(self) -> dict[str, ParallelStyle]:
        raise NotImplementedError("Subclasses must implement the layer_cp_plan property to define the context parallelism plan for the model layers.")
    
    @property
    def log_dict(self):
        """Return the log dictionary for the current training or evaluation step."""
        mode = "train" if self.model.training else "eval"
        return self._metrics[mode]
    
    def check_fn(self, module: torch.nn.Module):
        raise NotImplementedError("Subclasses must implement the check_fn property to define which layers should be checkpointed.")
    
    def chunk_model(self, model: torch.nn.Module):
        raise NotImplementedError("Subclasses must implement the chunk_model property to define how the model should be chunked for pipeline parallelism.")
    
    def prepare_fsdp(self, model: PreTrainedModel|PipelineStage):
        raise NotImplementedError("Subclasses must implement the prepare_fsdp property to define how the model should be wrapped with FSDP.")
    
    def _wrap_model(self, model:PreTrainedModel, training:bool = True, dataloader=None):

        if training:
            model = deepcopy(model)
            if self.args.ep_size > 1:
                raise NotImplementedError("Expert parallelism is not implemented yet.")

            if self.args.cp_size > 1:
                for name, module in model.named_modules():
                    for pattern, cp_style in self.layer_cp_plan.items():
                        if fnmatch(name, pattern) and isinstance(cp_style, ParallelStyle):
                            cp_style._apply(module, self.mesh["cp"])

            if self.args.attention_implementation == "sdpa":
                # we should register forward hook to replace the attention mask, as sdpa does not support masking
                # Modified form accelerate/big_model.py#_attach_context_parallel_hook
                def _self_attn_pre_forward(module, module_args, module_kwargs):
                    if "attention_mask" in module_kwargs:
                        if not self.args.bypass_causal_mask_check:
                            assert self._is_attention_mask_causal(module_kwargs["attention_mask"]), "SDPA attention only supports causal mask."
                        module_kwargs["attention_mask"] = None
                        if inspect.signature(getattr(module, "forward", lambda : None)).parameters.get("is_causal", None) is not None:
                            module_kwargs["is_causal"] = True

                    return module_args, module_kwargs

                for name, module in model.named_modules():
                    if name.endswith("self_attn"):
                        module.register_forward_pre_hook(
                            _self_attn_pre_forward, with_kwargs=True,prepend=True
                        )
            elif self.args.attention_implementation != "eager":
                # we should mock the DTenosr stuff to make sure they are convert to tensor first
                def convert_dtensor_wrapper(func):
                    def wrapper(*args, **kwargs):
                        new_args = []
                        device_mesh = None
                        for arg in args:
                            if isinstance(arg, DTensor):
                                device_mesh = arg.device_mesh
                                arg = arg.full_tensor()
                            new_args.append(arg)
                        for k, v in kwargs.items():
                            if isinstance(v, DTensor):
                                device_mesh = v.device_mesh
                                kwargs[k] = v.full_tensor()
                        attn_output, attn_weights = func(*new_args, **kwargs)
                        
                        if device_mesh is not None:
                            attn_output = DTensor.from_local(
                                attn_output,
                                device_mesh=device_mesh,
                            ).redistribute(
                                placements=[Shard(1)]
                            ).to_local()
                        
                        return attn_output, attn_weights
                    return wrapper
                
                from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
                ALL_ATTENTION_FUNCTIONS[self.args.attention_implementation] = convert_dtensor_wrapper(ALL_ATTENTION_FUNCTIONS[self.args.attention_implementation])
            
            if self.args.tp_size > 1:
                model = parallelize_module(
                    model,
                    device_mesh=self.mesh["tp"],
                    parallelize_plan=self.layer_tp_plan,
                )

            if self.args.pp_size > 1:
                model = self.chunk_model(model)
                self.pp_schedule = IterSchedule1F1B(
                    stage=PipelineStage(
                        model, 
                        stage_index=self.pp_rank,
                        num_stages=self.args.pp_size,
                        device=self.accelerator.device,
                        group=self.pp_group
                    ),
                    n_microbatches=self.args.gradient_accumulation_steps,
                    loss_fn=self.compute_loss_func
                )

            # load model weights
            if self.args.cpu_ram_efficient_loading and (self.args.resume_from_checkpoint or self.args.model_name_or_path):
                model = efficient_loading(model if self.args.pp_size <= 1 else model.module, self.args, self.model_cfg, device=self.accelerator.device, full_state_dict=self.full_state_dict)
            else:
                logger.info(f"Training model from scratch.")
            
            # apply checkpoint if needed
            if self.args.gradient_checkpointing or self.accelerator.state.fsdp_plugin.activation_checkpointing:
                apply_activation_checkpointing(
                    model,
                    checkpoint_wrapper_fn=checkpoint_wrapper,
                    check_fn=self.check_fn,
                )
            if self.args.activation_offloading:
                if self.args.tp_size > 1:
                    from patches.offloading import offload_wrapper
                else:
                    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import offload_wrapper
                model = offload_wrapper(model)
            
            if self.args.torch_compile:
                if self.args.torch_compile_backend:
                    model = torch.compile(model, backend=self.args.torch_compile_backend, mode=self.args.torch_compile_mode)
                else:
                    model = torch.compile(model, mode=self.args.torch_compile_mode)
            self.prepare_fsdp(model)
            if self.rank == 0:
                logger.debug(str(model))

            return model
        else:
            # TODO: Support eval model wrapping
            raise NotImplementedError("Eval model wrapping is not implemented yet.")

    @contextlib.asynccontextmanager
    @torch.no_grad()
    async def infer(self, split: Literal["train", "eval"] = "train"):
        """Context manager to setup and teardown the inference service."""
        
        state_dict = self.gather_model_state_dict(self.model,device=torch.device("cpu"))
        # offload models onto cpu
        self.model.cpu()
        
        if self.optimizer.state:
            device_maps = []
            # offload optimizers onto cpu
            for param_group in self.optimizer.param_groups:
                device_map = {}
                for param in param_group["params"]:
                    state = self.optimizer.state[param]
                    for k, v in state.items():
                        if isinstance(v, (torch.Tensor, DTensor)):
                            device_map[k] = v.device
                            state[k] = v.to(device="cpu", non_blocking=True)
                        # elif isinstance(v, DTensor):
                        #     device_map[k] = (v.device_mesh,v.placements)
                        #     state[k] = v.cpu()

                device_maps.append(device_map)
        
        clear_device_cache()  # clear the device cache to free up memory
        dist.barrier()
        
        # setup the inference service
        await self.inf_manager.resume_memory_occupation()
        # update model weights
        await self.inf_manager.update_model_weights(state_dict=state_dict, device=self.accelerator.device)
        # await self.inf_manager.update_weights_from_dist(
        #     named_tensors=[(name, param) for name, param in state_dict.items()],
        #     device=self.accelerator.device
        # )
        del state_dict
        clear_device_cache(True)  # clear the device cache to free up memory
        logger.info(f"Inference started, waiting for sampling tasks.")
        dist.barrier()
        
        split_lock = await DistributedLock.create(name=split)
        await split_lock.reset()

        sem = await DistributedCounter.create(name=f"infer")
        await sem.inc(1)

        yield self
        
        await sem.dec(1)
        await sem.wait_for(0, option="eq")

        await split_lock.set()
        if self.args.sync_sampling:
            running_sampling = await DistributedCounter.create(name=f"running-sampler-{split}")
            await running_sampling.wait_for(0, option="eq")

        logger.info(f"Inference finished, releasing resources.")
        await self.inf_manager.release_memory_occupation()

        dist.barrier()
        self.model = self.model.to(self.accelerator.device, non_blocking=True)
        
        # load the optimizer and model back to the device
        if self.optimizer.state:
            for gid,param_group in enumerate(self.optimizer.param_groups):
                device_map = device_maps[gid]
                for param in param_group["params"]:
                    state = self.optimizer.state[param]
                    for k, v in state.items():
                        if isinstance(v, (torch.Tensor, DTensor)):
                            device = device_map[k]
                            state[k] = v.to(device=device, non_blocking=True)
                        # elif isinstance(v, DTensor):
                        #     device_mesh, placements = device_map[k]
                        #     state[k] = state[k].to(device_mesh=device_mesh, placements=placements)
                        
        dist.barrier()

    def unified_loss_calculation(self, outputs, labels: DataLabels, num_items_in_batch=None):
        """A unified loss calculation function that handles different loss functions based on the configuration.

        Args:
            outputs: The model outputs.
            labels (DataLabels): The labels for the data.
            num_items_in_batch (int, optional): Number of items in the batch. Defaults to None.

        Returns:
            The calculated loss.
        """
        labels = DataLabels(**labels) if not isinstance(labels, DataLabels) else labels
        if self.args.cp_size > 1:
            with torch.no_grad():
                # we should mannually chunck labels along the sequence dimension for context parallelism
                labels.target_ids = labels.target_ids.chunk(self.args.cp_size, dim=1)[self.cp_rank]
                labels.advantage_mask = labels.advantage_mask.chunk(self.args.cp_size, dim=1)[self.cp_rank]
                labels.assistant_mask = labels.assistant_mask.chunk(self.args.cp_size, dim=1)[self.cp_rank]
                labels.create_step_mask = labels.create_step_mask.chunk(self.args.cp_size, dim=1)[self.cp_rank]
                if labels.per_token_logprobs is not None:
                    labels.per_token_logprobs = labels.per_token_logprobs.chunk(self.args.cp_size, dim=1)[self.cp_rank]

        match self.args.loss_calculater.lower():
            case "grpo" | "gspo" | "minirl":
                loss = self.grpo_loss(outputs, labels, num_items_in_batch)
            case "crossentropy":
                loss = self.cross_entropy_loss(outputs, labels, num_items_in_batch,)
            case _:
                raise ValueError(f"Unsupported loss_calculater: {self.args.loss_calculater}. Supported values are 'GRPO' and 'CrossEntropy'.")
        
        with torch.no_grad():
            mask = labels.create_step_mask != 0
            sampled_at_step = labels.create_step_mask[mask].float().mean().item() if mask.any() else 0.0
            self._local_log(
                {
                    "rewards": labels.rewards,
                    "/length/forward": labels.target_ids.size(1) * labels.target_ids.size(0),
                    "/length/context": (labels.target_ids != -100).sum(dim=1).cpu().tolist(),
                    "/length/completion": labels.assistant_mask.sum(dim=1).cpu().tolist(),
                    "scores": labels.scores,
                    "steps": labels.steps,
                    "advantages": labels.advantages,
                    "sampled_at_step": sampled_at_step,
                }
            )
        assert torch.isnan(loss) == False, "Loss is NaN detectted for labels: {}".format(labels)
        return loss

    def cross_entropy_loss(self, outputs, labels: DataLabels, num_items_in_batch=None, ignore_index: int = -100,):
        logits = outputs.logits if isinstance(outputs, dict) else outputs
        assert torch.isnan(logits).sum() == 0, "Logits contain NaN values"
        with torch.no_grad():
            # set non assitant part to ignore_index
            target = labels.target_ids * labels.assistant_mask + (1 - labels.assistant_mask) * ignore_index
            loss_tokens_count = (target != -100).sum()

            # convert target as DTensor if logits is DTensor
            if isinstance(logits, DTensor):
                target = distribute_tensor(
                    target,
                    device_mesh=logits.device_mesh,
                    placements=[Replicate() for _ in range(len(logits.placements))]
                )
            
            if self.args.cp_size > 1:
                dist.all_reduce(loss_tokens_count, op=dist.ReduceOp.SUM, group=self.cp_group)

        if self.args.loss_seq_chunk_size is None:
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1),  # (B * Seq, Vocab)
                target.flatten(0, 1).long(), # (B * Seq,)
                ignore_index=ignore_index,
                reduction="sum"
            )
        else:
            loss = 0
            for i in range(0, logits.size(1), self.args.loss_seq_chunk_size):
                loss = loss + torch.nn.functional.cross_entropy(
                    logits[:, i:i+self.args.loss_seq_chunk_size].flatten(0, 1),  # (B * Seq, Vocab)
                    target[:, i:i+self.args.loss_seq_chunk_size].flatten(0, 1).long(), # (B * Seq,)
                    ignore_index=ignore_index,
                    reduction="sum"
                )
        
        # loss to tensor if it is a DTensor
        if isinstance(loss, DTensor):
            loss = loss.full_tensor()
        # logger.debug("loss: {}".format(loss/loss_tokens_count.clamp_min(1)))
        if self.args.cp_size > 1:
            loss = torch.distributed.nn.functional.all_reduce(loss, op=dist.ReduceOp.SUM, group=self.cp_group)
        loss = loss / loss_tokens_count.clamp_min(1)

        with torch.no_grad():
            assistant_mask = labels.assistant_mask  # remove the first token assistant_mask
            if self.args.log_entropy:
                token_entropy = self.compute_token_entropy(logits)  # (B, T)
                entropy = (token_entropy * assistant_mask).sum() / assistant_mask.sum().clamp_min(1)
                entropy = entropy.item()
                self._local_log({
                    "loss_token_count_per_batch": loss_tokens_count.item(),
                    "entropy": entropy,
                })
        
        return loss

    def grpo_loss(self, outputs, labels: DataLabels, num_items_in_batch=None):
        logits = outputs.logits if isinstance(outputs, dict) else outputs
        advantage_mask = labels.advantage_mask
        assistant_mask = labels.assistant_mask
        create_step_mask = labels.create_step_mask
        target_ids = labels.target_ids
        loss_mask = assistant_mask
        
        # NOTE: Avoid materializing full tensor for very long sequences when using TP sharded vocab.
        if isinstance(logits, DTensor):
            local_logits = logits.to_local()  # (B, T, V_local)
            logits_group = logits.device_mesh.get_group()

            vocab_global = logits.size(-1)
            vocab_local = local_logits.size(-1)
            vocab_per_shard = vocab_global // logits.device_mesh.size(-1)

            local_start = logits.device_mesh.get_local_rank(-1) * vocab_per_shard
            local_end = local_start + vocab_local

            # mask marking tokens belonging to this shard
            valid = (target_ids >= local_start) & (target_ids < local_end)

            # pull local-target logits safely
            local_indices = (target_ids - local_start).clamp(0, vocab_local - 1)
            local_target_logits = torch.gather(
                local_logits, dim=-1, index=local_indices.unsqueeze(-1)
            ).squeeze(-1)

            # invalid tokens must contribute 0; only the shard owning the token contributes
            local_target_logits = local_target_logits * valid

            # 1. all_reduce sum → global target logits
            target_logits = torch.distributed.nn.functional.all_reduce(
                local_target_logits,
                op=dist.ReduceOp.SUM,
                group=logits_group
            ) # (B, T-1)

            # ----------------------------------------------------------------------
            # 2. Cross-shard logsumexp
            # ----------------------------------------------------------------------

            # per-shard max logits
            local_max = torch.max(local_logits, dim=-1).values  # (B, T)

            # global max over shards
            global_max = torch.distributed.nn.functional.all_reduce(
                local_max,
                op=dist.ReduceOp.MAX,
                group=logits_group
            ).detach()  # (B, T)

            # exp-sum for this shard
            local_exp = torch.exp(local_logits - global_max.unsqueeze(-1)).sum(dim=-1)  # (B, T)

            # global exp-sum
            global_exp = torch.distributed.nn.functional.all_reduce(
                local_exp,
                op=dist.ReduceOp.SUM,
                group=logits_group
            )

            # final global logsumexp
            global_lse = torch.log(global_exp) + global_max  # (B, T)
            # ----------------------------------------------------------------------
            # 3. final per-token log probabilities
            # ----------------------------------------------------------------------
            per_token_logps = target_logits - global_lse
        elif isinstance(logits, torch.Tensor):
            # 1. extract target logits
            target_logits = logits.gather(-1, target_ids.unsqueeze(-1).clamp(0, logits.size(-1) - 1)).squeeze(-1)  # (B, T)
            # 2. compute logsumexp
            lse = torch.logsumexp(logits, dim=-1)   # (B, T)
            # 3. compute per-token log-prob
            per_token_logps = target_logits - lse
        else:
            raise ValueError("logits must be either torch.Tensor or DTensor.")

        # Compute the policy ratio and clipped version
        if labels.per_token_logprobs is not None:
            old_per_token_logps = labels.per_token_logprobs.detach()
        else:
            old_per_token_logps = per_token_logps.detach()

        match self.args.loss_calculater.lower():
            case "grpo":
                coef_1 = torch.exp(per_token_logps - old_per_token_logps)
                coef_2 = torch.clamp(coef_1, 1 - self.args.epsilon, 1 + self.args.epsilon + self.args.epsilon_higher)

            case "gspo":
                coef_1 = torch.exp(((per_token_logps - old_per_token_logps)*assistant_mask).sum(dim=1) / assistant_mask.sum(dim=1))
                coef_1 = per_token_logps / per_token_logps.detach() * coef_1.detach()
                coef_2 = torch.clamp(coef_1, 1 - self.args.epsilon, 1 + self.args.epsilon + self.args.epsilon_higher)
            case "minirl":
                coef_1 = torch.exp(per_token_logps - old_per_token_logps)
                coef_2 = coef_1  # not used in minirl
                
            case _:
                raise NotImplementedError("Unsupported loss {}".format( self.args.loss_calculater))

        if self.args.strict_in_bound:
            decay_factor = 1 / self.args.importance_weight_cap_ratio
        else:
            if self.args.out_of_date_decay is not None:
                # calculate the out-of-date decay factor, using exponential decay
                decay_factor =  (create_step_mask - self.current_step) * self.args.out_of_date_decay
                decay_factor = torch.exp(torch.clamp(decay_factor, max=0.0))
            else:
                decay_factor = 1.0

        upper_bound = 1 + (self.args.epsilon + self.args.epsilon_higher) * self.args.importance_weight_cap_ratio * decay_factor
        lower_bound = 1 - self.args.epsilon * self.args.importance_weight_cap_ratio * decay_factor
        # modify loss mask to zeros where out of bound
        out_of_bound_mask = ((coef_1 > upper_bound) & (advantage_mask > 0)) | ((coef_1 < lower_bound) & (advantage_mask < 0))
        loss_mask = loss_mask * (~out_of_bound_mask).float()

        if self.args.loss_calculater.lower() in ["minirl"]:
            per_token_loss = - coef_1.detach() * advantage_mask * per_token_logps
        else:
            per_token_loss1 = coef_1 * advantage_mask
            per_token_loss2 = coef_2 * advantage_mask
            
            if self.args.beta1 is not None:
                low_mask = (coef_1 < 1 - self.args.epsilon) & (advantage_mask < 0)
                per_token_loss2[low_mask] = self.args.beta1 * (1 - self.args.epsilon) / coef_1.detach()[low_mask] * advantage_mask[low_mask]
            if self.args.beta2 is not None:
                high_mask = (coef_1 > 1 + self.args.epsilon + self.args.epsilon_higher) & (advantage_mask > 0)
                per_token_loss2[high_mask] = self.args.beta2 * (1 + self.args.epsilon + self.args.epsilon_higher) / coef_1.detach()[high_mask] * advantage_mask[high_mask]

            per_token_loss = - torch.min(per_token_loss1, per_token_loss2)
        
        if not self.args.token_level_loss:
            if self.args.skip_length_normalization:
                loss = (per_token_loss * loss_mask).sum(dim=1).mean()
            else:
                loss = ((per_token_loss * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp_min(1)).mean()
        else:
            local_numerator = (per_token_loss * loss_mask).sum(dim=1)
            local_denominator = loss_mask.sum(dim=1)
            global_denominator = torch.distributed.nn.functional.all_reduce(
                local_denominator,
                op=dist.ReduceOp.SUM,
            )
            global_numerator = torch.distributed.nn.functional.all_reduce(
                local_numerator,
                op=dist.ReduceOp.SUM,
            )
            loss = (global_numerator / global_denominator.clamp_min(1)).mean()

        with torch.no_grad():
            log_dict = {}
            # Log clip ratio
            if self.args.loss_calculater.lower() not in ["minirl"]:
                is_clipped = (per_token_loss1 < per_token_loss2).float()
                clip_ratio = (is_clipped * assistant_mask).sum() / assistant_mask.sum().clamp_min(1)
                clip_ratio = clip_ratio.item()
                log_dict["clip_ratio"] = clip_ratio
            if self.args.log_entropy:
                token_entropy = self.compute_token_entropy(logits)  # (B, T)
                entropy = (token_entropy * assistant_mask).sum() / assistant_mask.sum().clamp_min(1)
                entropy = entropy.item()
                log_dict["entropy"] = entropy
            if self.args.out_of_date_decay is not None or self.args.strict_in_bound:
                log_dict["out_of_date_ratio"] = ((out_of_bound_mask.float() * assistant_mask).sum() / assistant_mask.sum().clamp_min(1)).item()
            self._local_log(log_dict)
        
        return loss

    def compute_token_entropy(self, logits):
        """
        Compute per-token entropy without constructing softmax/log_softmax.
        Supports both:
            - Non-Tensor-Parallel logits: (B, T, V)
            - DTensor TP-sharded logits: (B, T, V_local)

        Args:
            logits: Tensor or DTensor (B, T, V or V_local)

        Returns:
            entropy_per_token: Tensor (B, T)
        """

        chunk_size = self.args.loss_seq_chunk_size
        T = logits.size(1)
        if chunk_size is None or chunk_size <= 0 or chunk_size >= T:
            ranges = [(0, T)]
        else:
            ranges = [(s, min(s + chunk_size, T)) for s in range(0, T, chunk_size)]

        # ------------------------------
        # Case 1: Regular tensor (no TP)
        # ------------------------------
        if not isinstance(logits, DTensor):
            ent_chunks: list[torch.Tensor] = []
            for start, end in ranges:
                lg = logits[:, start:end]  # (B, Tc, V)
                lse = torch.logsumexp(lg, dim=-1)  # (B, Tc)
                m = lg.max(dim=-1, keepdim=True).values.detach()
                exp_shifted = torch.exp(lg - m)  # (B, Tc, V)
                Z = exp_shifted.sum(dim=-1)  # (B, Tc)
                weighted_z = (exp_shifted * lg).sum(dim=-1)  # (B, Tc)
                ent_chunks.append(lse - weighted_z / Z)
            return ent_chunks[0] if len(ent_chunks) == 1 else torch.cat(ent_chunks, dim=1)

        # ------------------------------
        # Case 2: DTensor (TP sharded)
        # ------------------------------
        local_logits = logits.to_local()  # (B, T, V_local)
        logits_group = logits.device_mesh.get_group()
        ent_chunks: list[torch.Tensor] = []
        for start, end in ranges:
            ll = local_logits[:, start:end]  # (B, Tc, V_local)
            # Local maximum
            local_max = ll.max(dim=-1).values  # (B, Tc)
            # Global maximum
            global_max = torch.distributed.nn.functional.all_reduce(
                local_max, op=dist.ReduceOp.MAX, group=logits_group
            ).detach() # (B, Tc)
            # Stable shift + exp
            exp_local = torch.exp(ll - global_max.unsqueeze(-1))  # (B, Tc, V_local)
            local_Z = exp_local.sum(dim=-1)  # (B, Tc)
            local_weighted_z = (exp_local * ll).sum(dim=-1)  # (B, Tc)
            # Global reduction
            Z = torch.distributed.nn.functional.all_reduce(
                local_Z, op=dist.ReduceOp.SUM, group=logits_group
            )  # (B, Tc)
            weighted_z = torch.distributed.nn.functional.all_reduce(
                local_weighted_z, op=dist.ReduceOp.SUM, group=logits_group
            )  # (B, Tc)
            lse = global_max + torch.log(Z.clamp_min(1e-12))  # (B, Tc)
            ent_chunks.append(lse - weighted_z / Z.clamp_min(1e-12))
        return ent_chunks[0] if len(ent_chunks) == 1 else torch.cat(ent_chunks, dim=1)

    def _local_log(self, logs: dict[str, float]) -> None:
        for k,v in logs.items():
            if isinstance(v, Iterable):
                self.log_dict[k].extend(v)
            else:
                self.log_dict[k].append(v)

    async def compute_record_score_metrics(self):
        metrics = {}

        # all scores
        pipeline_all = [
            {"$match": {
                "split": "train",
                "status": {"$in": ["scored", "ready","abandoned"]},
            }},
            {
                "$group": {
                    "_id": None,
                    "mean": {"$avg": "$score"},
                    "max": {"$max": "$score"},
                    "min": {"$min": "$score"},
                    "std": {"$stdDevPop": "$score"},
                }
            },
            {"$project": {
                "_id": 0,
                "mean": 1,
                "max": 1,
                "min": 1,
                "std": 1,
            }},
        ]

        result_all = await Record.aggregate(pipeline_all).to_list()

        if result_all:
            r = result_all[0]
            metrics.update({
                "/record/score/mean": r.get("mean") or 0.0,
                "/record/score/max": r.get("max") or 0.0,
                "/record/score/min": r.get("min") or 0.0,
                "/record/score/std": r.get("std") or 0.0,
            })


        pipeline_last5 = [
            {
                "$match": {
                    "split": "train",
                    "status": {"$in": ["scored", "ready","abandoned"]},
                    "$or": [
                        {"last_trained_step": {"$gte": max(0, self.state.global_step - 5)}},
                        {"last_trained_step": -1}
                    ]
                }
            },
            {
                "$group": {
                    "_id": None,
                    "mean": {"$avg": "$score"},
                    "max": {"$max": "$score"},
                    "min": {"$min": "$score"},
                    "std": {"$stdDevPop": "$score"},
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "mean": 1,
                    "max": 1,
                    "min": 1,
                    "std": 1
                }
            }
        ]

        result_last5 = await Record.aggregate(pipeline_last5).to_list()

        if result_last5:
            r = result_last5[0]
            metrics.update({
                "/record/score/last_5_mean": r.get("mean") or 0.0,
                "/record/score/last_5_max": r.get("max") or 0.0,
                "/record/score/last_5_min": r.get("min") or 0.0,
                "/record/score/last_5_std": r.get("std") or 0.0,
            })
            
        return metrics


    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        if self.rank == 0:
            logs.update(self.loop.run_until_complete(self.compute_record_score_metrics()))

        log_dicts = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(log_dicts, self.log_dict if self.tp_rank == 0 else None)
        log_dict = defaultdict(list)
        for ld in log_dicts:
            if ld is not None:
                for key, val in ld.items():
                    if isinstance(val, (list, tuple)):
                        filtered_list = list(filter(lambda x: x is not None, val))
                        if len(filtered_list) != len(val):
                            logger.warning(f"Warning: {key} has {len(val)} values, but {len(filtered_list)} values after filtering.")
                        log_dict[key].extend(filtered_list)
                    else:
                        if val is None:
                            logger.warning(f"Warning: {key} has None value.")
                            continue
                        log_dict[key].append(val)
                        

        metrics = {key: sum(val) / len(val) if len(val) > 0 else 0 for key, val in log_dict.items()}  # average the metrics

        for key, val in log_dict.items():
            if isinstance(val, (list,tuple)) and len(val) > 1:
                metrics[key+'/max'] = max(val)
                metrics[key+'/min'] = min(val)
                metrics[key+'/std'] = sum(map(lambda x: (x - metrics[key]) ** 2, val)) ** 0.5 / len(val)
            elif isinstance(val, torch.Tensor):
                metrics[key+'/max'] = val.max().item()
                metrics[key+'/min'] = val.min().item()
                metrics[key+'/std'] = val.std().item()
        
        if not self.model.training:
            metrics = {f"eval_{k}": v for k, v in metrics.items()}
        
        logs = {**logs, **metrics}
        
        if "loss" in logs:
            # we should mannually fix the loss scale with gradiant accumulation
            logs["loss"] = logs["loss"] / self.args.gradient_accumulation_steps
        
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self.log_dict.clear()

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        
        # For DBIterableDataset we already collate inside the dataset; use batch_size=1 and identity collate
        if isinstance(train_dataset, DBIterableDataset):
            dataloader_params["batch_size"] = 1
            dataloader_params["collate_fn"] = (lambda batch: batch[0])
        
        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor # This is useless for num_workers=0

        return DataLoader(train_dataset, **dataloader_params)

    async def _async_sampling(self, epoch_iterator, num_batches):
        ctx = self.infer() if self.args.enable_sampling else contextlib.nullcontext()
        batch_samples = []
        global_step_counter = await DistributedCounter.create(name="global_step")
        if self.rank == 0:
            await global_step_counter.set({"n": self.state.global_step + 1})
        await global_step_counter.sync()
        self.current_step = global_step_counter.n

        async with ctx:
            for _ in range(num_batches):
                try:
                    item = next(epoch_iterator)
                    if item is not None:
                        batch_samples.append(item)
                except StopIteration:
                    break

        return batch_samples
    
    def get_batch_samples(self, epoch_iterator, num_batches, device):
        num_items_in_batch = None
        
        s_time = time.time()
        batch_samples = self.loop.run_until_complete(self._async_sampling(epoch_iterator, num_batches))

        # sync items across parallel groups with minimal payload
        if self.args.tp_size > 1:
            batch_samples = auto_broadcast(batch_samples, device=self.accelerator.device, group=self.tp_group)
        if self.args.cp_size > 1:
            batch_samples = auto_broadcast(batch_samples, device=self.accelerator.device, group=self.cp_group)
        if self.args.pp_size > 1:
            batch_samples = auto_broadcast(batch_samples, device=self.accelerator.device, group=self.pp_group)
        # clean up the batch samples to avoid potential memory leak
        gc.collect()
        torch.cuda.empty_cache()
        
        batch_count = len(batch_samples)
        if self.is_dp_master:
            logger.debug(f"Collected {batch_count} batch samples for training in {time.time() - s_time:.2f} seconds.")
        
        if batch_count < num_batches:
            logger.warning(f"Only {batch_count} samples collected, expected {num_batches}.")
            # replicate the batch_samples to num_batches
            # if batch_count > 0:
                # batch_samples = batch_samples * (num_batches // batch_count) + batch_samples[:num_batches % batch_count]

        all_batch_counts = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(all_batch_counts, batch_count)
        if any(map(lambda x: x == 0, all_batch_counts)):
            # If any process has no samples, we need to drop all samples
            logger.warning("Some processes have no samples, dropping all samples and stop training.")
            batch_samples.clear()
            self.control.should_training_stop = True

        count_num_items_in_batch = (
            len(batch_samples) > 0
            and batch_samples[0] is not None
            and isinstance(batch_samples[0], dict)
            and "labels" in batch_samples[0]
            and (
                # num_items_in_batch is passed to model forward
                # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer.py#L3757
                self.model_accepts_loss_kwargs
                # num_items_in_batch is passed to compute_loss_func
                # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer.py#L3773
                or self.compute_loss_func is not None
                # num_items_in_batch is also verified if (self.model_accepts_loss_kwargs or self.compute_loss_func)
                # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer.py#L3790
            )
        )

        if count_num_items_in_batch:
            # For now we don't support object detection
            try:
                num_items_in_batch = sum([(batch["labels"].ne(-100)).sum() for batch in batch_samples])
            except (TypeError, AttributeError):
                pass

        if num_items_in_batch is not None:
            if self.args.average_tokens_across_devices:
                num_items_in_batch = self.accelerator.gather(num_items_in_batch).sum()

            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(device)

                if self.args.n_gpu > 1 and num_items_in_batch.dim() == 0:
                    # In the DataParallel case, convert the scalar tensor into a 1-dim tensor
                    num_items_in_batch = num_items_in_batch.unsqueeze(0)

        return batch_samples, num_items_in_batch

    def gather_model_state_dict(self, model: torch.nn.Module, device: Optional[torch.device] = None) -> dict[str,torch.Tensor]:
        """
        Gather the model state dict from all ranks.
        Only the rank 0 of pp group will return the full state dict.
        Other ranks will return an empty dict.
        """
        local_state_dict = model.state_dict()
        state_dict = {}
        
        # sync the params source between pp ranks
        if self.args.pp_size > 1:
            param_src_list = [(name, self.pp_rank) for name in local_state_dict.keys()]
            gather_list = [None for _ in range(self.args.pp_size)]
            dist.all_gather_object(gather_list, param_src_list, group=self.pp_group)
            param_src = {}
            for param_src_list in gather_list:
                for name, src in param_src_list:
                    param_src[name] = src
        else:
            param_src = {name: self.pp_rank for name in local_state_dict.keys()}
        
        if self.rank == 0:
            pbar = tqdm.tqdm(total=len(self.global_state_dict), desc="Gathering model parameters",)

        for param_name, param_tensor in self.global_state_dict.items():
            if param_name in local_state_dict:
                tensor: torch.Tensor = local_state_dict[param_name]
                if isinstance(tensor, DTensor):
                    tensor = tensor.full_tensor()
                tensor = tensor.detach()
                # logger.debug(f"Gathering parameter {param_name} from tp group took {time.time() - stime:.2f}s")
            else:
                tensor = None
            
            # transfer the tensor to the pp rank 0
            if tensor is None:
                tensor = torch.empty_like(param_tensor, dtype=self.torch_dtype, device=torch.device("cuda"))
            dist.broadcast(tensor, group=self.pp_group, group_src=param_src[param_name])

            # Only the global rank 0 consumes the gathered tensor for saving
            if self.rank == 0:
                state_dict[param_name] = tensor.to(device)
                pbar.update(1)
                pbar.set_description(f"Gathering model parameters ({param_name})")
            else:
                state_dict[param_name] = torch.empty_like(param_tensor, dtype=self.torch_dtype, device=torch.device("meta"))

        return state_dict
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Will save the model, so you can reload it using `from_pretrained()`.

        Will only save from the main process.
        """

        if output_dir is None:
            output_dir = self.args.output_dir
            
        state_dict = self.gather_model_state_dict(self.model, device=torch.device("cpu"))
        if self.args.should_save:
            os.makedirs(output_dir, exist_ok=True)
            with torch.device("meta"):
                _model_to_save = self.model_class(self.model_cfg).to(dtype=self.torch_dtype)
            _model_to_save.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors)
            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
            # also save processor
            if self.processing_class is not None and hasattr(self.processing_class, "save_pretrained"):
                self.processing_class.save_pretrained(output_dir)
        
        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save", revision=self.args.hub_revision)
        
    def clean(self):
        """Clean up resources."""
        if hasattr(self, "inf_manager") and self.inf_manager is not None:
            self.inf_manager.clean()
        logger.info("AsyncGRPOTrainer cleaned up successfully.")

    def get_total_train_batch_size(self, args) -> int:
        """Calculates total batch size (micro_batch * grad_accum * dp_world_size).

        Note: Only considers DP and TP (dp_world_size = world_size // tp_size)."""
        dp_world_size = self.mesh["dp"].size()
        return self._train_batch_size * args.gradient_accumulation_steps * dp_world_size
    
    def _inner_training_loop(
        self, batch_size=None, args: AgentTrainingConfig=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            raise NotImplementedError("auto_find_batch_size is not implemented yet.")
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self.get_total_train_batch_size(args)

        (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)

        num_train_tokens = None
        if self.args.include_tokens_per_second:
            num_train_tokens = self.num_tokens(train_dataloader, None if epoch_based else max_steps)
            # If going by epochs, multiply tokens linearly
            if len_dataloader is not None and epoch_based:
                num_train_tokens *= args.num_train_epochs
            # Otherwise since its steps, we just multiply by grad accum
            else:
                num_train_tokens *= args.gradient_accumulation_steps

        from transformers.trainer import TrainerState,ExportableState
        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        self.state.compute_steps(args, max_steps)

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)
        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model
        
        self.create_optimizer_and_scheduler(args.max_steps)

        # ckpt loading
        # if resume_from_checkpoint is not None:
            # self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        self._load_scaler(resume_from_checkpoint)

        if self.tp_rank == 0 and self.dp_rank == 0:
            logger.info(f"  Optimized Params:\n{[]}")
            # Collect parameter names actually managed by optimizer (listed by param_groups)
            opt_param_ids = {id(p) for g in self.optimizer.param_groups for p in g["params"]}
            optimized_params = [name for name, p in model.named_parameters() if id(p) in opt_param_ids]
            # For readability, group output (each group corresponds to param_groups)
            group_outputs = []
            for gi, g in enumerate(self.optimizer.param_groups):
                g_names = [name for name, p in model.named_parameters() if id(p) in {id(x) for x in g["params"]}]
                group_outputs.append(f"[Group {gi} | wd={g.get('weight_decay','')} lr={g.get('lr','')}]:\n  " + "\n  ".join(g_names))
            logger.info("  Optimized Params (total {}):\n{}\n".format(len(optimized_params), "\n\n".join(group_outputs)))

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        from transformers.trainer_pt_utils import get_model_param_count
        from transformers.trainer import TRAINER_STATE_NAME,skip_first_batches,TrainOutput
        # Train!
        if self.rank == 0:
            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {num_examples:,}")
            logger.info(f"  Num Epochs = {num_train_epochs:,}")
            logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
            if self.args.per_device_train_batch_size != self._train_batch_size:
                logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
            logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
            logger.info(f"  Total optimization steps = {max_steps:,}")
            logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

    
        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        for attr in ("model", "optimizer", "lr_scheduler"):
            setattr(self.callback_handler, attr, getattr(self, attr))
        self.callback_handler.train_dataloader = train_dataloader

        self.state.init_training_references(self, max_steps, num_train_epochs, trial)

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0, device=args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        learning_rate = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        for epoch in range(epochs_trained, num_train_epochs):
            epoch_dataloader = train_dataloader
            if hasattr(epoch_dataloader, "set_epoch"):
                epoch_dataloader.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                # epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                self._load_rng_state(resume_from_checkpoint)
            
            step = -1
            epoch_iterator = iter(epoch_dataloader)
            # We chunkify the epoch iterator into gradient accumulation steps `n` batches
            remainder = steps_in_epoch % args.gradient_accumulation_steps
            if remainder == 0:
                remainder = args.gradient_accumulation_steps
            update_step = -1
            total_updates = steps_in_epoch // args.gradient_accumulation_steps + int(
                remainder < args.gradient_accumulation_steps
            )
            for _ in range(total_updates):
                update_step += 1
                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, args.device)
                # Store the number of batches for current gradient accumulation
                # This is used to correctly scale the loss when the last accumulation step has fewer batches
                self.current_gradient_accumulation_steps = len(batch_samples)
                
                if self.args.pp_size > 1 and len(batch_samples) > 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    # modify and prepare the inputs
                    labels = [batch.pop("labels") for batch in batch_samples]
                    inputs = batch_samples
                    
                    # from torch.distributed.tensor.parallel import loss_parallel
                    # with loss_parallel():
                    ctx = loss_parallel if self.args.enable_loss_parallel and self.args.tp_size != 1 else contextlib.nullcontext
                    attn_ctx = partial(sdpa_kernel,SDPBackend.FLASH_ATTENTION) if self.args.attention_implementation == "sdpa" else contextlib.nullcontext
                    with ctx(), attn_ctx():
                        losses = []
                        if self.pp_rank == self.args.pp_size - 1:
                            self.pp_schedule.step(
                                kwarg_mbs=inputs,
                                target_mbs=labels,
                                losses=losses,
                            )
                        else:
                            self.pp_schedule.step(
                                kwarg_mbs=inputs,
                                target_mbs=labels
                            )
                        
                    # sync losses accross ranks
                    if len(losses) > 0:
                        if isinstance(losses[0], DTensor):
                            losses = [loss.to_local() for loss in losses]
                        tr_loss = torch.mean(torch.stack(losses)).detach()
                    else:
                        tr_loss = None
                    # broad cast across pp ranks
                    tr_loss_list = [tr_loss]
                    dist.broadcast_object_list(tr_loss_list, group_src=self.args.pp_size - 1, group=self.pp_group)
                    tr_loss = tr_loss_list[0]
                    tr_loss = tr_loss.to(self.accelerator.device)
                    
                    self.accelerator.gradient_state._set_sync_gradients(True)
                    
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                    self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                    self.optimizer.step()

                    self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                    # get leaning rate before update
                    learning_rate = self._get_learning_rate()

                    if not self.accelerator.optimizer_step_was_skipped:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    self._maybe_log_save_evaluate(
                        tr_loss,
                        grad_norm,
                        model,
                        trial,
                        epoch,
                        ignore_keys_for_eval,
                        start_time,
                        learning_rate=learning_rate,
                    )
                    
                    continue
                
                for i, inputs in enumerate(batch_samples):
                    step += 1
                    do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
                    # Since we perform prefetching, we need to manually set sync_gradients
                    self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                    if self.args.include_num_input_tokens_seen:
                        main_input_name = getattr(self.model, "main_input_name", "input_ids")
                        if main_input_name not in inputs:
                            warnings.warn(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using.",
                                UserWarning, stacklevel=2,
                            )
                        else:
                            input_tokens = inputs[main_input_name].numel()
                            input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)
                            self.state.num_input_tokens_seen += self.accelerator.gather(input_tokens).sum().item()

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                    ctx = loss_parallel if self.args.enable_loss_parallel and self.args.tp_size != 1 else contextlib.nullcontext
                    attn_ctx = partial(sdpa_kernel,SDPBackend.FLASH_ATTENTION) if self.args.attention_implementation == "sdpa" else contextlib.nullcontext
                    with ctx(), attn_ctx():
                        tr_loss_step = self.training_step(model, inputs, num_items_in_batch)

                    if (
                        args.logging_nan_inf_filter
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        if tr_loss.device != tr_loss_step.device:
                            raise ValueError(
                                f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                            )
                        tr_loss = tr_loss + tr_loss_step

                    self.current_flos += float(self.floating_point_ops(inputs))
                    del inputs

                    if do_sync_step:
                        # Since we perform prefetching, we need to manually set sync_gradients to True
                        self.accelerator.gradient_state._set_sync_gradients(True)

                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                        self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                        context = contextlib.nullcontext
                        if self.is_tp_enabled:
                            from torch.distributed.tensor.experimental import implicit_replication

                            context = implicit_replication

                        with context():
                            self.optimizer.step()

                        self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                        # get leaning rate before update
                        learning_rate = self._get_learning_rate()

                        if not self.accelerator.optimizer_step_was_skipped:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.lr_scheduler.step()

                        model.zero_grad()
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                        self._maybe_log_save_evaluate(
                            tr_loss,
                            grad_norm,
                            model,
                            trial,
                            epoch,
                            ignore_keys_for_eval,
                            start_time,
                            learning_rate=learning_rate,
                        )
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        break
                # We also need to break out of the nested loop
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(
                tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=learning_rate
            )

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    @torch.no_grad()
    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (Union[`Dataset`, dict[str, `Dataset`]), *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. If it is a dictionary, it will
                evaluate on each dataset, prepending the dictionary key to the metric name. Datasets must implement the
                `__len__` method.

                <Tip>

                If you pass a dictionary with names of datasets as keys and datasets as values, evaluate will run
                separate evaluations on each dataset. This can be useful to monitor how training affects other
                datasets or simply to get a more fine-grained evaluation.
                When used with `load_best_model_at_end`, make sure `metric_for_best_model` references exactly one
                of the datasets. If you, for example, pass in `{"data1": data1, "data2": data2}` for two datasets
                `data1` and `data2`, you could specify `metric_for_best_model="eval_data1_loss"` for using the
                loss on `data1` and `metric_for_best_model="eval_data2_loss"` for the loss on `data2`.

                </Tip>

            ignore_keys (`list[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        self.model.eval()
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        start_time = time.time()
        
        async def run_eval():
            running_scheduler = await DistributedCounter.create("eval-running")
            async with self.infer(split="eval"):
                await running_scheduler.wait_for(0, "gt")
                await running_scheduler.wait_for(0, "eq")

            # all sampler finished, calculate the metrics
            logger.info("All evaluation sampling tasks completed.")
            
            metrics = {}
            global_step_counter = await DistributedCounter.create(name="global_step")
            ret = await Task.find_all(with_children=True).aggregate(
                [
                    {"$match": {"split": "eval", "added_step": global_step_counter.n}},
                    {
                        "$group": {
                            "_id": None,
                            "total": {"$sum": 1},
                            "scores": {"$push": {
                                "$avg": "$scores"
                            }},
                        }
                    },
                ]
            ).to_list()
            
            if ret:
                ret = ret[0]
                metrics["eval_total_tasks"] = ret["total"]
                ret["scores"] = [s for s in ret["scores"] if s is not None]
                metrics["eval_scores"] = sum(ret["scores"]) / len(ret["scores"]) if len(ret["scores"]) > 0 else 0.0
            else:
                metrics["eval_total_tasks"] = 0
                metrics["eval_scores"] = 0.0

            return metrics
        
        metrics = self.loop.run_until_complete(run_eval())
        dist.barrier()
        # TODO: calculate some metrics here
        metrics.update(speed_metrics(
            metric_key_prefix,
            start_time,
            metrics["eval_total_tasks"],
        ))
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        self._memory_tracker.stop_and_update_metrics(metrics)
        self.model.train()  # switch back to training mode
        return metrics
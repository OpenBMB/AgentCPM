import torch
import os
import socket
import base64
import contextlib
import torch.distributed as dist

from collections import Counter
from datetime import timedelta
from typing import Optional, Union, Any, Mapping
from PIL import Image
from io import BytesIO
from difflib import SequenceMatcher
from transformers import PretrainedConfig, AutoProcessor, AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
from torch.distributed.tensor import DTensor, distribute_tensor
from accelerate import init_empty_weights

from log import logger
from configs import AgentTrainingConfig
from databases import RecordData, DataLabels

def _compute_tensor_nbytes(t: torch.Tensor) -> int:
    try:
        return t.numel() * t.element_size()
    except Exception:
        # Fallback in unexpected cases
        return int(t.nelement()) * int(getattr(t, "element_size", lambda: 1)())



def setup_model_and_processor(training_cfg:AgentTrainingConfig):
    override_cfg = {}
    if training_cfg.decoder_sparse_step is not None:
        override_cfg["decoder_sparse_step"] = training_cfg.decoder_sparse_step

    cfg = AutoConfig.from_pretrained(
        training_cfg.model_name_or_path,
        attn_implementation=training_cfg.attention_implementation,
        dtype=getattr(torch, training_cfg.torch_dtype, torch.bfloat16),
        trust_remote_code=training_cfg.trust_remote_code,
        **override_cfg
    )
    
    ctx = init_empty_weights() if training_cfg.cpu_ram_efficient_loading else contextlib.nullcontext()
    with ctx:
        if training_cfg.trainer == "QwenVLTrainer":
            model = AutoModelForImageTextToText.from_config(cfg, trust_remote_code=training_cfg.trust_remote_code)
        else:
            model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=training_cfg.trust_remote_code)
    full_state_dict = None

    if training_cfg.cpu_ram_efficient_loading and dist.get_rank() == 0:
        model_name_or_path = training_cfg.resume_from_checkpoint if training_cfg.resume_from_checkpoint else training_cfg.model_name_or_path
        if training_cfg.trainer == "QwenVLTrainer":
            full_state_dict = AutoModelForImageTextToText.from_pretrained(
                model_name_or_path,
                attn_implementation=training_cfg.attention_implementation,
                dtype=getattr(torch, training_cfg.torch_dtype, torch.bfloat16),
                trust_remote_code=training_cfg.trust_remote_code,
                **override_cfg
            ).state_dict()
        else:
            full_state_dict = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                attn_implementation=training_cfg.attention_implementation,
                dtype=cfg.dtype,
                trust_remote_code=training_cfg.trust_remote_code,
                **override_cfg
            ).state_dict()
        logger.info("Model weight loaded from {}.".format(model_name_or_path))


    processing_class = AutoProcessor.from_pretrained(
        training_cfg.model_name_or_path,
        trust_remote_code=training_cfg.trust_remote_code,
    )
    dist.barrier()

    return model, processing_class, full_state_dict

@torch.no_grad()
def _convert_data_into_inputs_labels(
    data: list[RecordData],
    processor: AutoProcessor,
    max_length: int,
    pad_to_multiple_of: int = 0,
    shift_labels: bool = True,
    **kwargs,
) -> dict:
    """Convert following list of data into inputs and labels in a dict for training.
    {
        "messages": [{...},{...}],
        "tools": [{...},{...}], # or None
        "scores": {response_index: score}, # index for response message
        "advantages": {response_index: advantage}, # float number
        "reward": 0.3, # float number
    }

    Return:
        A dict with "labels" key and other model forwarding keys.
    """
    # Step 1: tokenize input messages with tools one by one
    # assert the role of the last message must be assistant
    for item in data:
        assert item.messages[-1]["role"] == "assistant", "The last message role MUST be assistant!"

    if not hasattr(processor, "tokenizer"):
        setattr(processor, "tokenizer", processor)

    match processor.__class__.__name__:
        case "Qwen3VLProcessor":
            from transformers import Qwen3VLProcessor
            processor: Qwen3VLProcessor = processor
            # For Qwen3VLProcessor, we should convert type:image_url into type:image before apply chat template
            for item in data:
                for message in item.messages:
                    if isinstance(message["content"],str):
                        continue
                    for content in message["content"]:
                        assert isinstance(content,dict), "list of content must be dict."
                        match content["type"]:
                            case "text":
                                pass
                            case "image_url":
                                content["type"] = "image"
                                content["image"] = content.pop("image_url")["url"]
                                if kwargs.get("max_pixels",None) is not None:
                                    content["max_pixels"] = kwargs["max_pixels"]
                            case "image":
                                if kwargs.get("max_pixels",None) is not None:
                                    content["max_pixels"] = kwargs["max_pixels"]
                            case "video_url":
                                content["type"] = "video"
                                content["video"] = content.pop("video_url")["url"]
                                if kwargs.get("max_pixels",None) is not None:
                                    content["max_pixels"] = kwargs["max_pixels"]
                            case "video":
                                if kwargs.get("max_pixels",None) is not None:
                                    content["max_pixels"] = kwargs["max_pixels"]
                            case _:
                                raise NotImplementedError("Content Type {} not supported for Qwen3VLProcessor.".format(content["type"]))
            
            text = []
            images = []
            videos = []
            from qwen_vl_utils import process_vision_info

            for item in data:
                # print(item.messages)
                text.append(
                    processor.apply_chat_template(
                        item.messages,
                        tools=item.tools,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                )
                imgs,vids = process_vision_info(item.messages, image_patch_size=processor.image_processor.patch_size)
                images.append(imgs)
                if vids is not None and len(vids) > 0:
                    videos.append(vids)
            
            inputs = processor(
                text=text,
                images=images,
                videos=videos if len(videos) > 0 else None,
                do_resize=False,
                return_tensors="pt",
                max_length=max_length,
                truncation=False, # FIXME: set to False to avoid cutting image/video tokens; maybe need a better way to handle truncation
                padding=True
            )
            index_pair = (
                processor.tokenizer.convert_tokens_to_ids('<|im_start|>'),
                processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
            )

        case "MiniCPMVProcessor":
            # For MiniCPMVProcessor, it cannot process content with list, we must convert them into strings before further process.
            images = []
            for item in data:
                imgs = []
                for message in item.messages:
                    if isinstance(message["content"],str):
                        continue
                    new_content_str = []
                    for content in message["content"]:
                        match content["type"]:
                            case "text":
                                new_content_str.append(content["text"])
                            case "image_url":
                                new_content_str.append("(<image>./</image>)")
                                assert not content["image_url"]["url"].startswith(("http","file")), "Loading images in this stage is not allow!"
                                image = Image.open(BytesIO(
                                    base64.b64decode(
                                        content["image_url"]["url"].split(",")[-1]
                                    )
                                ))
                                max_pixels = content.get("max_pixels", kwargs.get("max_pixels", None))
                                if max_pixels is not None and image.width * image.height > max_pixels:
                                    scale = (max_pixels / (image.width * image.height)) ** 0.5
                                    new_size = (int(image.width * scale), int(image.height * scale))
                                    image = image.resize(new_size, Image.Resampling.BICUBIC)
                                imgs.append(image)
                            case _:
                                raise NotImplementedError("Content Type {} not supported for MiniCPMVProcessor".format(content["type"]))
                    message["content"] = "\n".join(new_content_str)
                images.append(imgs)
            text = [
                processor.tokenizer.apply_chat_template(
                    item.messages,
                    tools=item.tools,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for item in data
            ]
            inputs = processor(
                text=text,
                images=images,
                return_tensors="pt",
                max_length=max_length,
                truncation=False, # FIXME: set to False to avoid cutting image/video tokens; maybe need a better way to handle truncation
            )
            index_pair = (
                processor.tokenizer.convert_tokens_to_ids('<|im_start|>'),
                processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
            )

        case _:
            text = [
                processor.apply_chat_template(
                    item.messages,
                    tools=item.tools,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for item in data
            ]
            inputs = processor(
                text=text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
            )
            index_pair = (
                processor.convert_tokens_to_ids('<|im_start|>'),
                processor.convert_tokens_to_ids('<|im_end|>')
            )

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = 0
    # Here, inputs are typically following dict:
    # {
    #   "input_ids":[[1,2,3,...],...],
    #   "attention_mask": [[1,1,1,...],...],
    #   # "pixel_values" or "image_grid_thw" something else for multimodal
    # }
    
    # Step 2: build assistant mask for inputs
    # divide input_ids into chunks according to index_pair
    assistant_mask = torch.zeros_like(inputs["input_ids"])
    advantage_mask = torch.zeros_like(inputs["input_ids"],dtype=torch.float)
    per_token_logprobs = torch.zeros_like(inputs["input_ids"],dtype=torch.float)
    create_step_mask = torch.zeros_like(inputs["input_ids"],dtype=torch.int)

    for batch_idx in range(inputs["input_ids"].size(0)):
        messages = data[batch_idx].messages
        logprobs = data[batch_idx].logprobs
        start_indices = torch.where(inputs["input_ids"][batch_idx] == index_pair[0])
        end_indices = torch.where(inputs["input_ids"][batch_idx] == index_pair[1])
        assert len(start_indices) == len(end_indices) and len(start_indices) == 1, "input_ids must be 2D shape!"
        start_indices = start_indices[0].tolist()
        end_indices = end_indices[0].tolist()
        if len(start_indices) != len(end_indices):
            logger.warning("Mismatched start and end indices in input_ids chunking! Possible truncation happens in record {}! Expected: {}, Got {} and {}. Token Nums: {}".format(data[batch_idx].record_id,len(messages), start_indices, end_indices, inputs["input_ids"][batch_idx].size()))
            if len(start_indices) == len(end_indices) + 1:
                # possible the last start index has no end index due to truncation, we add a end till the sequence end
                end_indices.append(inputs["input_ids"][batch_idx].size(0) -1)
            # make sure the length of start_indices and end_indices are aligned
            start_indices, end_indices = align_with_constant_gap(start_indices, end_indices)

        index_offset = min(max(len(start_indices) - len(messages), 0),1) # possible extra system prompt messages; 0 or 1
        input_ids_list = inputs["input_ids"][batch_idx].tolist()
        for origin_idx, advantage in data[batch_idx].advantages.items():
            created_at_step = data[batch_idx].created_at_step[origin_idx]
            idx = origin_idx + index_offset
            if idx >= len(start_indices):
                logger.warning("The message index {} exceed the start_indices  in record {}, possible due to truncation: Expect messages len {}, get {}".format(data[batch_idx].record_id, origin_idx, len(messages), len(start_indices)))
                continue
            
            if logprobs[origin_idx] is not None:
                tokens = [t["token"] for t in logprobs[origin_idx]["content"]]
                tids = processor.tokenizer.convert_tokens_to_ids(tokens)
                input_tids = input_ids_list[start_indices[idx] + 1: end_indices[idx] + 1]
                # find which part of the tids matches the input_ids
                matcher = SequenceMatcher(a=input_tids, b=tids)
                matched_blocks = matcher.get_matching_blocks()
                matched_length = 0
                for block in matched_blocks:
                    st = start_indices[idx] + 1 + block.a
                    ed = st + block.size
                    matched_length += block.size
                    
                    per_token_logprobs[batch_idx][st : ed] = torch.tensor([
                        t["logprob"] for t in logprobs[origin_idx]["content"][block.b:block.b + block.size]
                    ])
                    assistant_mask[batch_idx][st : ed] = 1 # ignore first start token
                    advantage_mask[batch_idx][st : ed] = advantage # set advantage
                    create_step_mask[batch_idx][st : ed] = created_at_step
                    
                if matched_length < min(len(tids), len(input_tids)):
                    logger.warning("Token logprobs length mismatch in record {} at message index {}: matched {}, input_ids {}, logprobs {}".format(
                        data[batch_idx].record_id,
                        origin_idx,
                        matched_length,
                        len(input_tids),
                        len(tids)
                    ))
            else:
                assistant_mask[batch_idx][start_indices[idx] + 1: end_indices[idx] + 1] = 1 # ignore first start token and add last end token
                advantage_mask[batch_idx][start_indices[idx] + 1: end_indices[idx] + 1] = advantage # set advantage
                create_step_mask[batch_idx][start_indices[idx] + 1: end_indices[idx] + 1] = created_at_step

    # Step 3: Shift labels if needed
    if shift_labels:
        # First token is shifted out, so we need to remove it from inputs
        target_ids = torch.nn.functional.pad(inputs["input_ids"][:, 1:], (0,1), value=-100)
        assistant_mask = torch.nn.functional.pad(assistant_mask[:, 1:], (0,1), value=0)
        advantage_mask = torch.nn.functional.pad(advantage_mask[:, 1:], (0,1), value=0)
        per_token_logprobs = torch.nn.functional.pad(per_token_logprobs[:, 1:], (0,1), value=0)
        create_step_mask = torch.nn.functional.pad(create_step_mask[:, 1:], (0,1), value=0)
    else:
        target_ids = inputs["input_ids"]
    
    # Step 4: padding to multiple of pad_to_multiple_of
    pad_length = inputs["input_ids"].size(1) % pad_to_multiple_of
    if pad_to_multiple_of > 0 and pad_length != 0:
        pad_length = pad_to_multiple_of - pad_length
        inputs["input_ids"] = torch.nn.functional.pad(inputs["input_ids"], (0, pad_length), value=pad_token_id)
        inputs["attention_mask"] = torch.nn.functional.pad(inputs["attention_mask"], (0, pad_length), value=0)
        target_ids = torch.nn.functional.pad(target_ids, (0, pad_length), value=-100)
        assistant_mask = torch.nn.functional.pad(assistant_mask, (0, pad_length), value=0)
        advantage_mask = torch.nn.functional.pad(advantage_mask, (0, pad_length), value=0)
        per_token_logprobs = torch.nn.functional.pad(per_token_logprobs, (0, pad_length), value=0)
        create_step_mask = torch.nn.functional.pad(create_step_mask, (0, pad_length), value=0)
        
    # # log content of assistant_mask for debugging
    # logger.debug("Assistant Content in Batch:" + str(processor.batch_decode(
    #     target_ids * assistant_mask,
    # )[0]))
    # Step 5: Adding position ids
    position_ids = inputs["attention_mask"].long().cumsum(-1) - 1
    position_ids.masked_fill_(inputs["attention_mask"] == 0, 1)
    inputs["position_ids"] = position_ids
    
    # Step 5: Prepare as final data batch
    all_scores = []
    all_advantages = []
    all_steps = []
    for item in data:
        all_scores.extend(item.scores.values())
        all_advantages.extend(item.advantages.values())
        all_steps.append(item.step)

    labels = {
        "scores": all_scores,
        "advantages": all_advantages,
        "rewards": [item.reward for item in data],
        "steps": all_steps,
        "target_ids": target_ids,
        "assistant_mask": assistant_mask,
        "advantage_mask": advantage_mask,
        "per_token_logprobs": None if torch.all(per_token_logprobs == 0) else per_token_logprobs,
        "create_step_mask": create_step_mask,
    }

    match processor.__class__.__name__:
        case "MiniCPMVProcessor":
            batch_data = {
                "data": {**inputs},
                "use_cache": False,
                "labels": labels
            }
        case _:
            batch_data = {
                **inputs,
                "use_cache": False,
                "labels": labels
            }
    batch_data = {**batch_data, **kwargs}
    return batch_data


def align_with_constant_gap(start_indices, end_indices):
    # Convert to list and sort (without changing original reference)
    starts = sorted(list(start_indices))
    ends = sorted(list(end_indices))

    # 1) Infer constant gap G: for each end find first start > end, collect gaps
    gaps = []
    si = 0
    for e in ends:
        # advance si to first > e
        while si < len(starts) and starts[si] <= e:
            si += 1
        if si < len(starts):
            gaps.append(starts[si] - e)

    if not gaps:
        logger.warning("Cannot infer constant gap G, no valid gaps found between starts and ends.")
        G = None
    else:
        # Take most common gap as G
        G = Counter(gaps).most_common(1)[0][0]

    # 2) Two pointers pair in order, enforce next_start - end == G (if G available)
    aligned_s = []
    aligned_e = []

    i, j = 0, 0  # i -> starts, j -> ends
    while i < len(starts) and j < len(ends):
        s = starts[i]
        # Skip ends that are before or equal to s
        if ends[j] <= s:
            j += 1
            continue

        # candidate end
        e = ends[j]

        # check condition: s < e always true here
        ok = False
        # Case A: If there is next start, require next_start - e == G (if G known)
        if i + 1 < len(starts):
            if G is None or (starts[i+1] - e == G):
                ok = True
        else:
            # Case B: s is the last start, allow pairing (no next_start to verify)
            ok = True

        if ok:
            aligned_s.append(s)
            aligned_e.append(e)
            i += 1
            j += 1
        else:
            # If gap not satisfied, decide which to discard—usually discard this end (too early end)
            # But can also discard current start; here choose to discard end (more conservative, avoid skipping start)
            j += 1

    return aligned_s, aligned_e

def efficient_loading(model: torch.nn.Module, args: AgentTrainingConfig, config: PretrainedConfig, device: torch.device, full_state_dict:dict) -> torch.nn.Module:
    
    model = fsdp2_load_full_state_dict_pipeline_aware(model, full_state_dict, device=device)
    return model


# Modified from accelerate.utils.fsdp_utils.fsdp2_load_full_state_dict
def fsdp2_load_full_state_dict_pipeline_aware(model: torch.nn.Module, full_sd: dict, device: torch.device = torch.device("cuda")):
    """
    Load a full state_dict (available only on rank 0) into a model that is:
      - partitioned by pipeline parallel (each rank has only some params),
      - possibly tensor-parallel via device_mesh / placements,
      - possibly FSDP/meta params.

    Args:
        model: the sharded model (params may be meta placeholders).
        full_sd: full state_dict present on rank 0 (None on other ranks).
    """
    world_rank = dist.get_rank()

    meta_sharded_sd = model.state_dict()  # local params/buffers only
    sharded_sd = {}
    # helper: decide casting/contiguous behavior (keep your original logic)
    def _infer_parameter_dtype(model: torch.nn.Module, param_name, empty_param):
        try:
            old_param = model.get_parameter_or_buffer(param_name)
        except AttributeError:
            base_param_name, local_param_name = param_name.rsplit(".", 1)
            submodule = model.get_submodule(base_param_name)
            old_param = getattr(submodule, local_param_name)

        is_torch_e4m3fn_available = hasattr(torch, "float8_e4m3fn")
        casting_dtype = None
        is_param_float8_e4m3fn = is_torch_e4m3fn_available and empty_param.dtype == torch.float8_e4m3fn

        if empty_param.dtype.is_floating_point and not is_param_float8_e4m3fn:
            casting_dtype = old_param.dtype

        return old_param is not None and old_param.is_contiguous(), casting_dtype

    def _cast_and_contiguous(tensor, to_contiguous, dtype):
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        if to_contiguous:
            tensor = tensor.contiguous()
        return tensor

    # --- 1) rank0 prepares metadata (ordered param list, shapes, dtypes) and broadcast to all ranks
    if world_rank == 0:
        if full_sd is None:
            raise RuntimeError("full_sd is None on rank 0 but expected to contain full model state_dict.")
        metadata = []
        for name, t in full_sd.items():
            metadata.append((name, tuple(t.size()), t.dtype))
        meta_obj = [metadata]
    else:
        meta_obj = [None]

    dist.broadcast_object_list(meta_obj, src=0)
    metadata = meta_obj[0]  # list of tuples (param_name, shape, dtype)

    # --- 2) iterate in the same global order and broadcast each full tensor (CPU to avoid GPU OOM by default)
    for param_name, shape, dtype in metadata:
        # Be careful: all ranks must be able to place a tensor on this device.
        # prepare a tensor to be broadcasted: rank0 uses full_sd[param_name], others allocate empty
        if world_rank == 0:
            full_param = full_sd[param_name].detach()
            # put on recv_device for consistent broadcast device
            try:
                full_param_bcast = full_param.to(device=device)
            except Exception:
                # fallback to cpu
                full_param_bcast = full_param.cpu()
            # ensure contiguous for broadcast
            full_param_bcast = full_param_bcast.contiguous()
        else:
            # allocate empty matching shape/dtype on recv_device
            full_param_bcast = torch.empty(size=shape, dtype=dtype, device=device)

        # perform broadcast (all ranks call this in same order)
        dist.broadcast(full_param_bcast, src=0)

        # Now each rank has the full param in full_param_bcast (on recv_device).
        # Only if this rank actually hosts this param (present in meta_sharded_sd) do we distribute it to sharded layout.
        if param_name in meta_sharded_sd:
            sharded_param = meta_sharded_sd[param_name]
            if isinstance(sharded_param, DTensor):
                # print(param_name, " is DTensor")
                device_mesh = sharded_param.device_mesh
                # distribute_tensor will create sharded tensor using device_mesh & placements
                try:
                    # If distribute_tensor expects the full tensor on CPU, consider .cpu()
                    # We pass full_param_bcast as-is; distribute_tensor will handle device movement.
                    sharded_tensor = distribute_tensor(full_param_bcast, device_mesh, sharded_param.placements)
                except Exception as e:
                    # fallback: move to CPU then try
                    logger.warning(f"distribute_tensor failed for {param_name} on device {full_param_bcast.device}: {e}. Retrying from CPU.")
                    sharded_tensor = distribute_tensor(full_param_bcast.cpu(), device_mesh, sharded_param.placements)

                # preserve dtype/contiguity expected by model
                to_contiguous, casting_dtype = _infer_parameter_dtype(model, param_name, full_param_bcast)
                sharded_tensor = _cast_and_contiguous(sharded_tensor, to_contiguous, casting_dtype)

                sharded_sd[param_name] = sharded_tensor

            elif isinstance(sharded_param, torch.Tensor):
                # print(param_name, " is Tensor")
                # regular param/buffer (not DTensor)
                if sharded_param.is_meta:
                    # meta placeholder: create a new tensor to hold the real data
                    sharded_param = torch.empty_like(full_param_bcast, device=device)
                else:
                    # non-meta param: we will copy into this existing tensor
                    if sharded_param.device != device:
                        # move to recv_device for consistent processing
                        sharded_param = sharded_param.to(device=device)

                # preserve dtype/contiguity expected by model
                to_contiguous, casting_dtype = _infer_parameter_dtype(model, param_name, sharded_param)
                full_param_bcast = _cast_and_contiguous(full_param_bcast, to_contiguous, casting_dtype)

                if sharded_param.shape != full_param_bcast.shape:
                    raise RuntimeError(f"Shape mismatch for param {param_name}: model expects {sharded_param.shape}, but full state_dict provides {full_param_bcast.shape}.")

                # copy full to local param
                sharded_param.copy_(full_param_bcast)
                sharded_sd[param_name] = sharded_param

            else:
                raise NotImplementedError()
        else:
            # param not local to this rank; skip storing it
            continue

    # --- 3) finally load into model (assign=True for meta placeholders)
    model.load_state_dict(sharded_sd, assign=True)
    return model

def obtain_local_port():
    def get_local_ip_to_master(master_addr):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((master_addr, 80))  # Connect without sending packets
            ip = s.getsockname()[0]
        finally:
            s.close()
        return ip

    local_ip = get_local_ip_to_master(os.environ.get('MASTER_ADDR',"localhost"))

    with socket.socket() as sock:
        sock.bind((local_ip, 0))
        local_ip, port = sock.getsockname()[:2]

    return local_ip,port

# ---- Recursive broadcast ----
def auto_broadcast(data, device, group, group_src=0):
    is_src_rank = dist.get_group_rank(group, dist.get_rank()) == group_src
    data_type_list = [type(data) if is_src_rank else None]
    dist.broadcast_object_list(data_type_list, group=group, group_src=group_src)
    data_type = data_type_list[0]

    if issubclass(data_type, torch.Tensor):
        # broadcast tensor metadata
        tensor_meta = [data.size(), data.dtype, data.layout, data.device] if is_src_rank else [None, None, None, None]
        dist.broadcast_object_list(tensor_meta, group=group, group_src=group_src)
        size, dtype, layout, _ = tensor_meta
        if not is_src_rank:
            data = torch.empty(size, dtype=dtype, layout=layout, device=device)
        else:
            data = data.to(device)
        dist.broadcast(data, group_src=group_src, group=group)
        return data.cpu() # move back to CPU to save GPU memory

    elif issubclass(data_type, (int, float, bool)) or data_type is type(None):
        obj = [data] if is_src_rank else [None]
        dist.broadcast_object_list(obj, group=group, group_src=group_src)
        return obj[0]

    elif issubclass(data_type, (list, tuple)):
        length = [len(data)] if is_src_rank else [0]
        dist.broadcast_object_list(length, group=group, group_src=group_src)
        length = length[0]
        return [auto_broadcast(data[i] if is_src_rank else None, device, group, group_src)
                for i in range(length)]

    elif issubclass(data_type, Mapping):
        keys = list(data.keys()) if is_src_rank else []
        length = [len(keys)] if is_src_rank else [0]
        dist.broadcast_object_list(length, group=group, group_src=group_src)
        n_keys = length[0]
        new_dict = {}
        for i in range(n_keys):
            key = [keys[i]] if is_src_rank else [None]
            dist.broadcast_object_list(key, group=group, group_src=group_src)
            key = key[0]
            new_dict[key] = auto_broadcast(data[key] if is_src_rank else None, device, group, group_src)
        return new_dict

    else:
        raise TypeError(f"Unsupported type: {type(data)}")

from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    Store,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)

# Copy from pytorch to allow creating multiple main groups.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py
def init_process_group(
    backend: Union[str, Backend] = None,
    init_method: Optional[str] = None,
    timeout: Optional[timedelta] = None,
    world_size: int = -1,
    rank: int = -1,
    store: Optional[Store] = None,
    group_name: str = None,
    pg_options: Optional[Any] = None,
):
    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)

    # NOTE: The pg_options parameter was renamed into backend_options in PyTorch 2.6.0
    # https://github.com/pytorch/pytorch/commit/a0c7029a75628cd5fa8df83c0de0ea98ee7fd844
    # We need to determine the appropriate parameter name based on PyTorch version
    from packaging import version
    pg_options_param_name = "backend_options" if version.parse(torch.__version__) >= version.parse("2.6") else "pg_options"
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{pg_options_param_name: pg_options},
        timeout=timeout,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg

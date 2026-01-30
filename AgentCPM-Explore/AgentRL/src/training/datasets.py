import asyncio
import torch.multiprocessing as mp
import base64
import aiohttp
import base64
import math
import mimetypes
import tenacity
import threading
import queue
import time

from beanie import UpdateResponse
from beanie.operators import In, Inc, Set, Or, NotIn
from typing import Optional, AsyncGenerator, Any, Callable
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase

from log import logger
from configs import AgentTrainingConfig
from databases import Record, DistributedLock, DistributedCounter, DispatchedSamplingTask, RecordData, init_databases, DBRecordData
from .utils import _convert_data_into_inputs_labels


class DBIterableDataset(IterableDataset):
    def __init__(
        self,
        args: AgentTrainingConfig,
        split: Optional[str] = "train",
        dp_rank: Optional[int] = 0,
        dp_size: Optional[int] = 1,
        fetching: Optional[bool] = True,
        convert_record_to_data_func: Optional[Callable[[Record, Optional[AgentTrainingConfig]], AsyncGenerator[RecordData, Any]]] = None,
        num_threads: int = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
    ):
        """Dataset that fetches records from the Mongo database.
        
        Args:
            loop: Optional event loop for asynchronous operations.
            split: Dataset split, can be "train", "valid", or "test". Defaults to "train".
        """
        
        super().__init__()
        self.args = args
        self.split = split
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.fetching = fetching
        self.skip_non_positive_advantage = args.skip_non_positive_advantage
        self.gradient_accumulation_steps = args.gradient_accumulation_steps if args.gradient_accumulation_steps is not None else 1
        self.batch_size = args.per_device_train_batch_size if args.per_device_train_batch_size is not None else 1
        self.convert_record_to_data_func = convert_record_to_data if convert_record_to_data_func is None else convert_record_to_data_func
        self.num_threads = num_threads if num_threads is not None else args.dataloader_num_workers
        self.processing_class = processing_class
        self.enable_sampling = args.enable_sampling
        self.max_trained_count = args.max_trained_count
        self.data_sem_name = str(self.split) + f'-fetch-{self.dp_rank}'

        # Use a process-safe queue when running background worker in a subprocess
        self.mp_ctx = mp.get_context("spawn")
        # Unified stop control via multiprocessing.Event to work with both processes and threads
        self._stop_event = self.mp_ctx.Event()
        
        # Start a subprocess to avoid interfering with main-thread asyncio
        self._start_bg_worker()

    def _start_bg_worker(self, resume=False):
        
        # Terminate existing worker if any
        if hasattr(self, "_bg_worker") and self._bg_worker is not None:
            try:
                self._bg_worker.terminate()
                self._bg_worker.join(timeout=1)
            except Exception:
                pass

        if self.fetching:
            self.fetch_cache = self.mp_ctx.Queue(maxsize=self.num_threads)
            self._bg_worker = self.mp_ctx.Process(target=self._bg_main, args=(resume,), daemon=True)
            self._bg_worker.start()
            logger.debug(f"Background fetch worker started with PID {self._bg_worker.pid}")

        self.collated_cache = queue.Queue(maxsize=self.args.dataloader_prefetch_factor)
        self._collate_thread = threading.Thread(target=self._collate_worker, daemon=True)
        self._collate_thread.start()

    def _bg_main(self, resume=False):
        """Background main with internal event-loop restart."""
        while True:  # supervise loop
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Reinitialize DB connection for this attempt
                loop.run_until_complete(init_databases(self.args))

                # Reset semaphore only when not resuming
                if not resume:
                    counter = loop.run_until_complete(
                        DistributedCounter.create(name=self.data_sem_name, n=0, group="fetching")
                    )
                    loop.run_until_complete(counter.reset())

                workers: list[asyncio.Task] = []
                # convertion worker
                concertion_task = loop.create_task(
                    self._convert_record_to_data()
                )
                workers.append(concertion_task)
                
                # fectching worker
                for i in range(self.num_threads):
                    fetch_task = loop.create_task(
                        self._fetching_record_data(fetch_idx=i)
                    )
                    workers.append(fetch_task)
                
                pending = set(workers)
                
                while pending:
                    done, pending = loop.run_until_complete(asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED))
                    for task in done:
                        if task.exception():
                            # cancel all pending tasks
                            for ptask in pending:
                                ptask.cancel()
                            raise task.exception()
                
                break  # exit supervise loop if completed normally

            except Exception as e:
                import traceback
                logger.error(
                    f"Async event loop crashed: {e}\n{traceback.format_exc()}\nRestarting loop..."
                )

            finally:
                try:
                    loop.close()
                except Exception:
                    pass

            # Short delay to avoid rapid crash loops
            time.sleep(1)
            resume = True   # After first run, always resume instead of reset

    def _collate_worker(self):

        def _collate_batch(samples: list[RecordData]):
            if len(samples) == 0 or any(s is None for s in samples):
                return None
            pad_to_multiple = (
                self.args.max_tokens
                if self.args.pp_size > 1 or self.args.pad_to_maximum
                else math.lcm(self.args.pad_to_multiple_of, self.args.tp_size * self.args.cp_size)
            )
            return _convert_data_into_inputs_labels(
                samples,
                processor=self.processing_class,
                max_length=self.args.max_tokens,
                shift_labels=self.args.shift_labels,
                pad_to_multiple_of=pad_to_multiple,
                output_router_logits=self.args.output_router_logits,
                max_pixels=self.args.max_pixels,
            )

        buffer = []
        previous_collated = None
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        data_sem = loop.run_until_complete(DistributedCounter.create(name=self.data_sem_name, group="fetching"))
        local_count = 1
        
        while not self._stop_event.is_set():
            if not self.fetching:
                loop.run_until_complete(data_sem.wait_for(local_count, option="gte"))
                self.collated_cache.put(None)
                local_count += 1
                continue
            
            buffer.append(self.fetch_cache.get())
            if len(buffer) < self.batch_size:
                continue
            raw_batch = buffer[:self.batch_size]
            buffer = buffer[self.batch_size:]
            try:
                collated = _collate_batch(raw_batch)
            except Exception as e:
                import traceback
                logger.error(f"Error while collating batch: {e}\n"+traceback.format_exc())
                collated = None
            
            if self.args.dynamic_batching and collated is not None and 'input_ids' in collated:
                seq_len = collated['input_ids'].shape[1]
                cur_bsz = len(raw_batch)

                # 如果当前 batch 的 bsz*len 还没达到阈值，尝试从队列中无阻塞地取更多样本并重打包
                while seq_len * cur_bsz < self.args.max_tokens:
                    try:
                        extra = None
                        extra = self.fetch_cache.get(timeout=1)
                    except queue.Empty:
                        break
                    if extra is None:
                        # 保持占位但不增加有效样本
                        buffer.append(extra)
                        break

                    raw_batch.append(extra)
                    # 重打包
                    try:
                        new_collated = _collate_batch(raw_batch)
                    except Exception as e:
                        import traceback
                        logger.error(f"Error while collating batch during dynamic batching: {e}\n"+traceback.format_exc())
                        new_collated = None
                    
                    if new_collated is None or 'input_ids' not in collated:
                        # 无法打包则撤回加入
                        buffer.insert(0, raw_batch.pop())
                        break

                    new_seq_len = new_collated['input_ids'].shape[1]
                    new_bsz = len(raw_batch)
                    if new_seq_len * new_bsz > self.args.max_tokens:
                        # 超过阈值，放弃新加入，使用之前较短的 batch
                        buffer.insert(0, raw_batch.pop())
                        break
                    else:
                        # 接受新打包，更新当前状态并记录上一个可行批次以便回退
                        previous_collated = collated
                        collated = new_collated
                        seq_len = new_seq_len
                        cur_bsz = new_bsz
            
            # 如果动态扩容导致不可用或超过阈值，则回退到之前的较短序列
            if collated is None and previous_collated is not None:
                self.collated_cache.put(previous_collated)
                previous_collated = None
            else:
                self.collated_cache.put(collated)
            
            loop.run_until_complete(data_sem.inc(1))

    async def _fetching_record_data(self, fetch_idx: int = 0):
        """Fetch a RecordData from the database."""
        logger.debug(f"Fetch thread {fetch_idx} for split {self.split} started.")
        data_sem = await DistributedCounter.create(name=self.data_sem_name, group="fetching")
        global_step_counter = await DistributedCounter.create(name="global_step")
        
        while not self._stop_event.is_set():
            await asyncio.sleep(0.1) 
            # check whether current fetcher should proceed
            await global_step_counter.sync()
            if await data_sem.check(
                global_step_counter.n * self.gradient_accumulation_steps,
                "gt"
            ):
                # check whether is minal counter
                minal_count = await DistributedCounter.find(
                    DistributedCounter.group == "fetching"
                ).min(DistributedCounter.n)
                if await data_sem.check(
                    minal_count,
                    "gt"
                ):
                    await asyncio.sleep(1) # wait other fetchers to catch up
                    continue
            
            # fetch a DBRecordData
            record_data: DBRecordData | None = await DBRecordData.find_one(
                DBRecordData.fetched == False,
                DBRecordData.split == self.split,
                with_children=True
            ).update(
                Set({DBRecordData.fetched: True}),
                response_type=UpdateResponse.NEW_DOCUMENT
            )
            if record_data is None:
                continue # not find
            
            data = record_data.to_record_data()
            await record_data.delete()
            data.messages = await preprocess_mm_messages_for_sample(data.messages)
            self.fetch_cache.put(data)
            
        self.fetch_cache.put(None)

    def __iter__(self):
        self._stop_event.clear()
        while True:
            item = self.collated_cache.get()
            yield item
    
    def stop_fetching(self):
        """Stop the background fetching thread."""
        if hasattr(self, "_stop_event") and self._stop_event is not None:
            self._stop_event.set()
        if hasattr(self, "_bg_worker") and self._bg_worker is not None:
            self._bg_worker.join()

    async def _sample_and_update_record(self, max_retry = 5) -> Optional[Record]:
        """Random sample a Record that meets the criteria and atomically update its trained_count and last_trained_step.
        """
        global_step_counter = await DistributedCounter.create(name="global_step")
        for _ in range(max_retry):
            await global_step_counter.sync()
            # 聚合随机抽样一个候选
            pipeline = [
                {
                    "$match": {
                        "trained_count": {"$lt": self.max_trained_count},
                        "$or": [
                            {"last_trained_step": -1},
                            {"last_trained_step": {"$lte": global_step_counter.n - self.args.retrained_interval}}
                        ],
                        "status": {"$in": [Record.Status.READY, Record.Status.ABANDONED]},
                        "split": self.split,
                        "score": {"$ne": None}
                    }
                },
                {"$sample": {"size": 1}},
            ]
            sampled = await Record.find_all(with_children=True).aggregate(pipeline,).to_list()
            if not sampled:
                return None
            candidate = sampled[0]
            candidate_id = candidate["_id"] if isinstance(candidate, dict) else candidate.id
            if candidate_id is None:
                return None
            # 原子更新（再次确认条件仍成立）
            updated = await Record.find_one(
                Record.id == candidate_id,
                Record.trained_count < self.max_trained_count,
                Or(
                    Record.last_trained_step == -1,
                    Record.last_trained_step <= global_step_counter.n - self.args.retrained_interval,
                ),
                In(Record.status, [Record.Status.READY, Record.Status.ABANDONED]),
                Record.split == self.split,
                Record.score != None,
                with_children=True
            ).update(
                Inc({Record.trained_count: 1}),
                Set({Record.last_trained_step: global_step_counter.n}),
                response_type=UpdateResponse.NEW_DOCUMENT
            )
            if updated is not None:
                return updated
            # 否则重试
            await asyncio.sleep(1)
        return None
    
    async def _convert_record_to_data(
        self,
    ):
        """Convert a Record object to a DBRecordData."""
        logger.debug(f"Convert worker for split {self.split} started.")
        epoch = await DistributedCounter.create(name=f"{self.split}-epoch")
        while not self._stop_event.is_set():
            record = await self._sample_and_update_record()
            if record is None:
                if self.enable_sampling:
                    if await epoch.check(int(self.args.num_train_epochs), option="lte"):
                        continue
                    else:
                        # check if there are any sampling tasks still running
                        runing_scheduler = await DistributedCounter.create(f"{self.split}-running")
                        if await runing_scheduler.check(0, option="gt"):
                            continue
                else:
                    if await epoch.check(int(self.args.num_train_epochs), option="lte"):
                        lock = await DistributedLock.create(name=f"epoch-update-{self.split}",)
                        if (await lock.set()):
                            await epoch.inc(1)
                            logger.info(f"Starting epoch {epoch.n} for split {self.split}.")
                            await Record.find(
                                Record.status == Record.Status.READY,
                                Record.split == self.split,
                                with_children=True
                            ).update(
                                {
                                    "$set": {
                                        Record.trained_count: 0,
                                        Record.last_trained_step: -1
                                    }
                                }
                            )
                            logger.info(f"Reset trained_count for all records in split {self.split}.")
                            await lock.reset()
                        else:
                            await lock.wait()
                        continue
                break
            
            try:
                async for data in self.convert_record_to_data_func(
                    record,
                    args=self.args
                ):
                    # convert to DBRecordData
                    new_data = DBRecordData.from_record_data(data)
                    await new_data.insert()
                    
            except Exception as e:
                import traceback
                logger.error(f"Error while converting record {record.id} to data: {e}\n"+traceback.format_exc())
                record.status = Record.Status.FAILED
                await record.save()
        
        if await epoch.check(int(self.args.num_train_epochs), option="gt"):
            logger.info(f"All epochs completed for split {self.split}. Stopping fetch.")
        else:
            logger.info(f"Fetch worker for split {self.split} stopping.")
        self._stop_event.set()

async def convert_record_to_data(
    record:Record,
    args: AgentTrainingConfig = None
) -> AsyncGenerator[RecordData, Any]:
    """Convert a Record object to a data dictionary."""
    task = await record.task.fetch(True)
    samples: list[DispatchedSamplingTask] = []
    for item in record.traj:
        sample = await item.fetch(True)
        if sample.status != sample.Status.COMPLETED:
            logger.debug(f"Skipping sample {sample.id} due to status {sample.status}.")
            continue
        samples.append(sample)

    # try to merge samples according to the prefix
    unique_samples = []
    data_queue: list[RecordData] = []
    step = len(samples)
    for sample in reversed(samples):
        step -= 1
        contained = False
        for idx, usample in enumerate(unique_samples):
            if is_contained_in_prefix(sample, usample):
                contained = True
                break
        
        if sample.advantage is not None:
            advantage = sample.advantage
            score = sample.score if sample.score is not None else record.score
        else:
            if sample.score is not None:
                advantage = sample.score
                score = sample.score
            else:
                advantage = record.score - (sum(task.scores) / len(task.scores))
                score = record.score

        response_index = len(sample.request["messages"])
        if not contained:
            unique_samples.append(sample)
            data_queue.append(
                RecordData(
                    messages=sample.request["messages"] + [sample.response["choices"][0]["message"]],
                    tools=sample.request.get("tools",None),
                    scores={response_index: score},
                    advantages={response_index: advantage},
                    logprobs={response_index: sample.response["choices"][0].get("logprobs", None)},
                    reward=record.score,
                    step=step,
                    record_id=str(record.id),
                    created_at_step={response_index: sample.created_at_step}
                )
            )
        else:
            # data should be replaced
            data_queue[idx].messages[response_index] = sample.response["choices"][0]["message"]
            data_queue[idx].scores[response_index] = score
            data_queue[idx].advantages[response_index] = advantage
            data_queue[idx].logprobs[response_index] = sample.response["choices"][0].get("logprobs", None)
            data_queue[idx].created_at_step[response_index] = sample.created_at_step

    valid_data = []
    for data in data_queue:
        # remove indexed with advantage < minimal advantage
        removed_index = list(filter(
            lambda idx: data.advantages[idx] < args.minimal_advantage if args.minimal_advantage \
                else (args.drop_zero_advantage and data.advantages[idx] == 0), data.advantages.keys()))
        if removed_index:
            for idx in removed_index:
                data.scores.pop(idx)
                data.advantages.pop(idx)
                data.logprobs.pop(idx)
        
        if data.scores:
            valid_data.append(data)
    
    if valid_data:
        logger.debug("Find {} samples in record {}.".format(len(valid_data),record.id))
        for data in reversed(valid_data):
            yield data


async def preprocess_mm_messages_for_sample(messages):
    """ If there're images to be downloaded, download them and replace the image URLs with the downloaded images.
        Avoid downloading images in collator
    """
    image_tasks = []    # collect (msg_idx, content_idx, url)
    for msg_id, msg in enumerate(messages):
        content = [msg["content"]] if isinstance(msg["content"], str) else  msg["content"]
        for c_idx, item in enumerate(content):
            if isinstance(item, dict) and item.get("type","") == "image_url":
                image_url: str = item["image_url"]["url"]
                if image_url.startswith("http"):
                    image_tasks.append((msg_id, c_idx, image_url))

    # download images concurrently
    if image_tasks:
        # 设置会话与请求超时
        client_timeout = aiohttp.ClientTimeout(total=15, connect=10, sock_read=15, sock_connect=10)
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            @tenacity.retry(
                wait=tenacity.wait_exponential(multiplier=1, min=2, max=15),
                stop=tenacity.stop_after_attempt(5),
                retry=(
                    tenacity.retry_if_exception_type(aiohttp.ClientError) |
                    tenacity.retry_if_exception_type(asyncio.TimeoutError)
                )
            )
            async def download_image_base64_from_url(url):
                # 对单个请求设置超时，避免长时间挂起
                req_timeout = aiohttp.ClientTimeout(total=15, connect=10, sock_read=15, sock_connect=10)
                async with session.get(url, timeout=req_timeout) as response:
                    response.raise_for_status()
                    image_bytes = await response.read()
                content_type = response.headers.get("Content-Type", "")
                if content_type.startswith("image/"):
                    mime = content_type.split(";")[0].strip()
                else:
                    mime = mimetypes.guess_type(url)[0] or "image/png"
                img_b64 = base64.b64encode(image_bytes).decode()
                return f"data:{mime};base64,{img_b64}"
            
            downloaded_images = await asyncio.gather(
                *[download_image_base64_from_url(url) for _,_,url in image_tasks],
                return_exceptions=True
            )

        # fill back images to messages
        for (msg_idx, content_idx, url), result in zip(image_tasks, downloaded_images):
            if not isinstance(result, Exception) and result is not None:
                messages[msg_idx]["content"][content_idx]["image_url"]["url"] = result
            else:
                raise result
    return messages

  
def is_same_context_message(message1, message2):
    """Check whether two messages have the same context.
    
    Possible keys in a message: role, content, audio, function_call, tool_calls, reasoning_content
    """
    for k in ["role", "content", "function_call", "tool_calls", "audio", "reasoning_content"]:
        if k not in message1 and k not in message2:
            continue

        if message1.get(k,None) is None:
            if message2.get(k,None) is not None:
                return False
            else:
                continue
        if message2.get(k,None) is None:
            if message1.get(k,None) is not None:
                return False
            else:
                continue

        match k:
            case "content":
                if isinstance(message1[k], str) and isinstance(message2[k], str):
                    if message1[k] != message2[k]:
                        return False
                    
                elif isinstance(message1[k], list) and isinstance(message2[k], list):
                    if len(message1[k]) != len(message2[k]):
                        return False
                    
                    for item1, item2 in zip(message1[k], message2[k]):
                        if isinstance(item1, str) and isinstance(item2, str):
                            if item1 != item2:
                                return False
                        elif isinstance(item1, dict) and isinstance(item2, dict):
                            if item1.get("type") != item2.get("type"):
                                return False
                            if item1.get("type") == "text":
                                if item1.get("text") != item2.get("text"):
                                    return False
                            elif item1.get("type") == "image_rul":
                                if item1.get("image_url", {}).get("url") != item2.get("image_url", {}).get("url"):
                                    return False
                            else:
                                return False
                        else:
                            return False
                else:
                    if message1.get(k,"") != message2.get(k,""):
                        return False

            case "function_call":
                function_call1 = message1.get(k, {})
                function_call2 = message2.get(k, {})
                if function_call1.get("name") != function_call2.get("name"):
                    return False
                args1 = function_call1.get("arguments", {})
                args2 = function_call2.get("arguments", {})
                if args1 != args2:
                    return False

            case "tool_calls":
                tool_calls1 = message1.get(k, [])
                tool_calls2 = message2.get(k, [])
                if len(tool_calls1) != len(tool_calls2):
                    return False
                for tc1, tc2 in zip(tool_calls1, tool_calls2):
                    if tc1.get("type") != tc2.get("type"):
                        return False
                    if tc1.get("function") != tc2.get("function"):
                        return False

            case "role" | "reasoning_content" | "audio":
                if message1.get(k,None) != message2.get(k,None):
                    return False
                
            case _:
                raise ValueError(f"Unknown key in message: {k}")
    return True


def is_contained_in_prefix(s1: DispatchedSamplingTask, s2: DispatchedSamplingTask):
    """check whether the sample1 is contained in sample2's prefix"""
    # Steps to check whether s1 is contained in s2's prefix:
    # 1. Check if s1 and s2 have the same tools list
    if "tools" in s1.request:
        if "tools" not in s2.request:
            return False
        if s1.request["tools"] != s2.request["tools"]:
            return False
    
    # 2. Check if the messages in s1 are a prefix of those in s2
    if len(s1.request["messages"]) >= len(s2.request["messages"]):
        return False
    for msg1, msg2 in zip(s1.request["messages"], s2.request["messages"]):
        if not is_same_context_message(msg1, msg2):
            return False
    # 3. Check if the responses in s1 are inside messages of s2
    if s1.response["choices"][0]["message"]:
        msg1 = s1.response["choices"][0]["message"]
        # it should be in the s2
        msg2 = s2.request["messages"][len(s1.request["messages"])]
        if not is_same_context_message(msg1, msg2):
            return False
    return True

"""
MCP-specific rewarding logic for task completion and record filtering.
"""
from log import logger
import random
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict
from beanie.operators import In
from beanie import UpdateResponse

if TYPE_CHECKING:
    from .models import MCPTask
    from databases import Record
    from configs import AgentTrainingConfig




async def apply_mcp_rewarding_logic(
    task: "MCPTask",
    record_class: type,
    task_class: type,
    config: Any
) -> None:
    """
    Apply MCP-specific rewarding logic similar to tool_rewarding.py.
    This includes variance check, balance_sample, filter_wrong, etc.
    
    Args:
        task: The MCPTask instance
        record_class: The Record class type
        task_class: The Task class type
        config: The AgentTrainingConfig instance
    """
    # Check if all scores are identical (variance check)
    # If max - min < 0.1, abandon the task as it provides no useful variance
    if max(task.scores) - min(task.scores) < 0.1:
        task.status = task_class.Status.COMPLETED
        await record_class.find_many(
            {"task.$id": task.id}, with_children=True
        ).update({"$set": {"status": record_class.Status.ABANDONED}})
        await task.save()
        return
    
    # Balance sample logic
    # This method will keep positive samples and negative samples in a balanced way.
    if getattr(config, "balance_sample", False):
        # Get wrong records (score == 0)
        pipeline = [
            {"$match": {"task.$id": task.id, "score": 0}},
            {"$addFields": {"traj_length": {"$size": "$traj"}}},
            {"$sort": {"traj_length": 1}}
        ]
        dict_records = await record_class.aggregate(pipeline).to_list()
        wrong_records = [record_class.model_validate(rec) for rec in dict_records]
        
        # Get right records (score != 0)
        pipeline = [
            {"$match": {"task.$id": task.id, "score": {"$ne": 0}}},
            {"$addFields": {"traj_length": {"$size": "$traj"}}},
            {"$sort": {"traj_length": 1}}
        ]
        dict_records = await record_class.aggregate(pipeline).to_list()
        right_records = [record_class.model_validate(rec) for rec in dict_records]
        
        # Filter out failed_error records
        valid_wrong_records = []
        for rec in wrong_records:
            final_answer = rec.meta_infos.get("final_answer", "")
            if final_answer == "failed_error":
                rec.status = record_class.Status.ABANDONED
                await rec.save()
            else:
                valid_wrong_records.append(rec)
        
        # Balance sampling
        min_len = min(len(valid_wrong_records), len(right_records))
        random.shuffle(valid_wrong_records)
        random.shuffle(right_records)
        
        for i, rec in enumerate(valid_wrong_records):
            if i < min_len + 1:
                rec.status = record_class.Status.READY
            else:
                rec.status = record_class.Status.ABANDONED
            await rec.save()
        
        for i, rec in enumerate(right_records):
            if i < min_len + 2:
                rec.status = record_class.Status.READY
            else:
                rec.status = record_class.Status.ABANDONED
            await rec.save()
        
        task.status = task_class.Status.COMPLETED
        await task.save()
        return
    
    # Default rewarding logic:
    # 1. Set all records to READY
    # 2. Abandon failed_error records to avoid training two many records encountered environment errors.
    
    # Step 1: Set all records to READY
    await record_class.find_many(
        {"task.$id": task.id}, with_children=True
    ).update({"$set": {"status": record_class.Status.READY}})
    
    all_records = await record_class.find_many(
        {"task.$id": task.id}, with_children=True
    ).to_list()
    all_records = [record_class.model_validate(rec) for rec in all_records]
    
    # Step 2: Abandon all failed_error records
    failed_error_records = [rec for rec in all_records if rec.meta_infos.get("final_answer", "") == "failed_error"]
    for rec in failed_error_records:
        rec.status = record_class.Status.ABANDONED
        await rec.save()
    
    # Step 2.5: Abandon correct samples (score > 0) with turns <= min_turns
    min_turns = getattr(config, "min_turns", 1)
    correct_records_to_abandon = []
    for rec in all_records:
        if rec.status == record_class.Status.ABANDONED:
            continue  # Skip already abandoned records
        score = rec.score if rec.score is not None else 0.0
        if score > 0:  # Correct samples
            turns_length = rec.meta_infos.get("turns", None)
            if turns_length is None:
                turns_length = len(rec.traj) if rec.traj else 0
            if turns_length <= min_turns:
                correct_records_to_abandon.append(rec)
    
    for rec in correct_records_to_abandon:
        rec.status = record_class.Status.ABANDONED
        await rec.save()
    
    if len(correct_records_to_abandon) > 0:
        logger.info(f"Abandoned {len(correct_records_to_abandon)} correct samples with turns <= {min_turns}.")
    
    #Step 3: Reserve necessary context_error records to avoid losing important information.
    # Randomly reserve a specified number of context error records (length_limit_error and turns_limit_error)
    # Only apply this logic when efficient_rewarding is False (manual control mode)
    if not getattr(config, "efficient_rewarding", False):
        context_error_types = ["length_limit_error", "turns_limit_error"]
        context_error_records = [rec for rec in all_records if rec.meta_infos.get("final_answer", "") in context_error_types]
        
        if len(context_error_records) > 0:
            context_error_records = [record_class.model_validate(rec) for rec in context_error_records]
            # Randomly shuffle and keep only the specified number
            random.shuffle(context_error_records)
            reserve_count = getattr(config, "reserve_context_error_count", 0)
            reserve_count = max(0, min(reserve_count, len(context_error_records)))  # Clamp to valid range
            
            for i, rec in enumerate(context_error_records):
                if i < reserve_count:
                    # Keep these records as READY (already set in Step 1)
                    pass
                else:
                    # Abandon the rest
                    rec.status = record_class.Status.ABANDONED
                    await rec.save()
            logger.info(f"Reserved {reserve_count} out of {len(context_error_records)} context error records.")
        
    # Step 4: Rank-Based Zero-Sum Adjustment
    # Only apply this logic when efficient_rewarding is True (efficient mode)
    if getattr(config, "efficient_rewarding", False):
        valid_records = [rec for rec in all_records if rec.status != record_class.Status.ABANDONED]
        
        if len(valid_records) > 0:
            score_0_records = []
            score_1_records = []
            
            for rec in valid_records:
                score = rec.score if rec.score is not None else 0.0
                turns = rec.meta_infos.get("turns", len(rec.traj) if rec.traj else 0)
                if score < 0.5:
                    score_0_records.append((rec, turns))
                else:
                    score_1_records.append((rec, turns))

            # ==============================================================
            # 通用核心函数：计算零和排名权重
            # ==============================================================
            async def apply_rank_zero_sum(records_list, total_spread, higher_turns_is_better):
                n = len(records_list)
                if n <= 1: return # 没法比较，不做调整
                
                # 1. 排序
                # 必须先按 turns 排序，才能正确处理相同排名的平均值
                sorted_records = sorted(records_list, key=lambda x: x[1])
                
                # 2. 计算平均排名 (Handle Ties)
                # ranks 数组存放每个样本的排名值
                ranks = [0.0] * n
                i = 0
                while i < n:
                    j = i
                    # 找到所有轮次相同的区间 [i, j)
                    while j < n and sorted_records[j][1] == sorted_records[i][1]:
                        j += 1
                    
                    # 计算该区间的平均排名
                    # 比如占据了位置 0, 1, 2，平均排名就是 1.0
                    avg_rank = sum(range(i, j)) / (j - i)
                    
                    for k in range(i, j):
                        ranks[k] = avg_rank
                    
                    i = j
                
                # 3. 计算排名的均值 (用于中心化)
                # 理论均值是 (n-1)/2，但在有 Tie 的情况下，算术平均值依然是最准的零和锚点
                mean_rank = sum(ranks) / n
                
                # 4. 计算步长 (Step Size)
                # 我们希望最极端的排名对应 +/- half_spread
                # adjustment = (rank - mean_rank) * step
                # max_deviation 约为 (n-1)/2
                # step = half_spread / ((n-1)/2) = spread / (n-1)
                step = total_spread / (n - 1) if n > 1 else 0
                
                # 5. 应用调整
                total_check = 0
                for idx, (rec, turns) in enumerate(sorted_records):
                    rank = ranks[idx]
                    
                    # 中心化：将排名变成围绕 0 的分布
                    centered_rank = rank - mean_rank 
                    
                    # 计算基础调整值
                    adjustment = centered_rank * step
                    
                    # 6. 决定方向
                    if not higher_turns_is_better:
                        # 如果希望轮次越短分越高 (如 Correct 组)
                        # 此时 Rank 越小(short)，centered_rank 是负数
                        # 我们需要它变正，所以取反
                        adjustment = -adjustment
                    else:
                        # 如果希望轮次越长分越高 (如 Wrong 组)
                        # Rank 越大(long)，centered_rank 是正数，直接加
                        pass
                    
                    old_score = rec.score if rec.score is not None else 0.0
                    rec.score = old_score + adjustment
                    await rec.save()
                    
                    total_check += adjustment
                    # logger.debug(f"Rec {rec.id}: turns={turns}, rank={rank}, adj={adjustment:+.4f}")
                
                return total_check

            # ==============================================================
            # 1. 错误组 (Wrong Group)
            # ==============================================================
            # 目标：Short(Rank小) -> 负分; Long(Rank大) -> 正分
            # 方向：Higher Turns is Better (in terms of penalty mitigation)
            if len(score_0_records) > 0:
                sum_check = await apply_rank_zero_sum(
                    score_0_records, 
                    total_spread=0.2,       # 范围 ±0.1
                    higher_turns_is_better=True 
                )
                try:
                    logger.info(
                        f"Wrong Group (Weighted Rank): N={len(score_0_records)}, "
                        f"Sum={0.0 if sum_check is None else sum_check:.4f}"
                    )
                except Exception as e:
                    logger.error(f"Error logging Wrong Group: {e}")

            # ==============================================================
            # 2. 正确组 (Correct Group)
            # ==============================================================
            # 目标：Short(Rank小) -> 正分; Long(Rank大) -> 负分
            # 方向：Higher Turns is Worse (Lower is Better)
            if len(score_1_records) > 0:
                sum_check = await apply_rank_zero_sum(
                    score_1_records, 
                    total_spread=0.2,       # 范围 ±0.1
                    higher_turns_is_better=False
                )
                try:
                    logger.info(
                        f"Correct Group (Weighted Rank): N={len(score_1_records)}, "
                        f"Sum={0.0 if sum_check is None else sum_check:.4f}"
                    )
                except Exception as e:
                    logger.error(f"Error logging Correct Group: {e}")
                    
    task.status = task_class.Status.COMPLETED
    await task.save()


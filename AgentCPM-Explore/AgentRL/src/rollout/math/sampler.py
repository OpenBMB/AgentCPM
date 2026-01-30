from ..sampler import StatefulSampler,AsyncSampler
from .utils import compute_score

from databases import Task, register_new_model, Record

import re

@register_new_model
class MATHTask(Task):
    problem: str
    answer: str
    subject: str
    level: int
    experience: str = "Solve the following math problem step-by-step."

class MATHSampler(AsyncSampler):
    async def run(self, task: MATHTask):
        messages = [
            {
                "role": "system",
                "content": "You are a highly intelligent math problem solver. Provide clear, step-by-step solutions to the problems presented."
            },
            {
                "role": "user",
                "content": "Answer the problem and give your final answers inside the \\boxed{}.\n"+task.problem
            }
        ]
        response = await self.create_chat_completions(
            messages=messages,
            temperature=0.7,
            repeat_penalty=1.2,
        )
        self.record.meta_infos["pred_ans"] = response.choices[0].message.content
    
    async def evaluate_record(self, record):
        """
        Calculate the score for the given record.

        Args:
            record (Record): The record to calculate the score for.
        Returns:
            float: The calculated score.
        """
        task: MATHTask = record.task
        gt_answer = task.answer
        try:
            is_correct, format_correctness, extracted_model_output = compute_score(record.meta_infos["pred_ans"], gt_answer)
            record.meta_infos["format_correctness"] = format_correctness
            record.meta_infos["extracted_model_output"] = extracted_model_output
        except:
            is_correct = False
        score = 1.0 if is_correct else 0.0
        return score

class MATHStateSampler(AsyncSampler):
    async def run(self, task: MATHTask):
        messages = [
            {
                "role": "system",
                "content": "You are a highly intelligent math problem solver. Provide clear, step-by-step solutions to the problems presented.\n<EXPERIENCE>\n"+task.experience+"\n</EXPERIENCE>"
            },
            {
                "role": "user",
                "content": "Answer the problem and give your final answers inside the \\boxed{}.\n"+task.problem
            }
        ]
        response = await self.create_chat_completions(
            messages=messages,
            temperature=0.7,
            repeat_penalty=1.2,
        )
        self.record.meta_infos["pred_ans"] = response.choices[0].message.content
        messages.append(response.choices[0].message.model_dump())
        messages.append({
            "role": "user",
            "content": "You have finished the task, now it's your turn to make a reflexion on what you learned to be important. Your updated experience should help you solve similar problems more effectively in the future. Remember to keep it concise and precise. Provide your updated experience in the format: <EXPERIENCE>...your experience...</EXPERIENCE>."
        })
        response = await self.create_chat_completions(
            messages=messages,
            temperature=0.7,
            repeat_penalty=1.2,
        )
        try:
            content = response.choices[0].message.content.split("</think>")[-1].strip()
            task.experience = re.search(r"<EXPERIENCE>(.*?)</EXPERIENCE>", content, re.DOTALL).group(1).strip()
        except Exception:
            task.experience = content
        await task.save()

    async def evaluate_record(self, record: Record) -> float:
        """
        Calculate the score for the given record.

        Args:
            record (Record): The record to calculate the score for.

        Returns:
            float: The calculated score.
        """
        # Simple handling here
        task: MATHTask = record.task
        gt_answer = task.answer
        try:
            is_correct, format_correctness, extracted_model_output = compute_score(record.meta_infos["pred_ans"], gt_answer)
            record.meta_infos["format_correctness"] = format_correctness
            record.meta_infos["extracted_model_output"] = extracted_model_output
        except:
            is_correct = False
        score = 1.0 if is_correct else 0.0
        # experience_sample = await record.traj[-1].fetch(True)
        # await experience_sample.save() # set default 0 advantage to avoid errors
        
        if record.traj_id > 0:
            # add score for experience update step
            for traj_id in range(record.traj_id):
                past_record = await Record.find_one({"task.$id": task.id, "traj_id": traj_id}, with_children=True)
                experience_sample = await past_record.traj[-1].fetch(True)
                if experience_sample.score is None:
                    experience_sample.score = 0.0
                experience_sample.score += (past_record.score - score) * 0.9**(record.traj_id - traj_id) if past_record.score is not None else 0.0
                experience_sample.advantage = experience_sample.score
                await experience_sample.save()
        return score
import random
from typing import Optional, Literal, Annotated
from beanie import Document, Indexed, UpdateResponse
import asyncio
import time
from log import logger
from pymongo.errors import DuplicateKeyError

class DistributedCounter(Document):
    """
    MongoDB-based distributed counter (root model).
    
    WARNING: Always using check to compare values!
    """
    # Unique identifier
    name: Annotated[str, Indexed(unique=True)]
    group: Annotated[str, Indexed(unique=False)] = "default"
    
    # Counter value
    n: int = 0  # this value is not stable!
    
    class Settings:
        is_root = True
    
    @classmethod
    async def create(cls, name: str, n: int = 0, group: str = "default") -> "DistributedCounter":
        """
        Create or Get a new semaphore instance with the given name.
        
        :param name: Unique name for the semaphore.
        :return: An instance of DistributedSemaphore.
        """
        doc = await cls.find_one({"name": name}, with_children=True)
        if doc is None:
            doc = cls(name=name, n=n, group=group)
            try:
                await doc.insert()
            except DuplicateKeyError:
                # Another process has already created the document, re-query
                await asyncio.sleep(0.1)  # Brief wait to ensure MongoDB commit completes
                doc = await cls.find_one({"name": name}, with_children=True)
                if doc is None:
                    # If still not found, might be other issue, re-raise exception
                    raise
            except Exception as e:
                # Other exception, try re-query once
                doc = await cls.find_one({"name": name}, with_children=True)
                if doc is None:
                    raise e
        return doc
    
    async def inc(self, amount: int = 1):
        """
        Increment the semaphore count.
        
        :param amount: The amount to increment the semaphore count by.
        """
        await DistributedCounter.find_one(
            {"name": self.name}, with_children=True
        ).update(
            {"$inc": {"n": amount}},
        )
        await self.sync()

    async def dec(self, amount: int = 1):
        """
        Decrement the semaphore count.
        
        :param amount: The amount to decrement the semaphore count by.
        """
        await DistributedCounter.find_one(
            {"name": self.name}, with_children=True
        ).update(
            {"$inc": {"n": -amount}},
        )
        await self.sync()
    
    async def reset(self):
        """
        Reset the semaphore count to zero.
        """
        doc = await DistributedCounter.find_one(
            {"name": self.name, "n": {"$ne": 0}}, with_children=True
        ).update(
            {"$set": {"n": 0}},
            response_type=UpdateResponse.NEW_DOCUMENT
        )
        if doc is None:
            logger.debug(f"Semaphore {self.name} is already reset.")
            await self.sync()
            return False
        logger.debug(f"Semaphore {self.name} is reset.")
        await self.sync()
        return True
        
    async def check(self, count: int, option: Literal["eq","gt","lt", "gte", "lte"]):
        """
        Check if the semaphore count meets the specified condition.
        """
        doc = await DistributedCounter.find_one({"name": self.name, "n": {f"${option}": count}}, with_children=True)
        return doc is not None

    async def wait_for(self, count: int, option: Literal["eq","gt","lt", "gte", "lte"], timeout: Optional[float] = None,):
        """
        Wait until the semaphore count is met the option of count using MongoDB Change Streams.
        
        Args:
            timeout: Timeout in seconds, None means wait indefinitely.
            
        Returns:
            True if count is met, False if timeout occurred
        """
        if await self.check(count, option):
            return True
        
        start_time = time.time()
        try:
            while True:
                if await self.check(count, option):
                    return True
                if timeout and (time.time() - start_time) >= timeout:
                    return False
                
                await asyncio.sleep(random.random()*0.2) # randomize
                
        except Exception as e:
            if "cannot schedule new futures after shutdown" in str(e):
                return False
            else:
                raise e


class DistributedLock(DistributedCounter):
    """
    MongoDB-based distributed lock derived from the counter root model.
    """

    async def set(self):
        """
        Set the semaphore, locking it for other threads.
        """
        doc = await DistributedLock.find_one(
            {"name": self.name, "n": 0}, with_children=True
        ).update(
            {"$set":{"n": 1}},
            response_type=UpdateResponse.NEW_DOCUMENT
        )
        return doc is not None
    
    async def is_locked(self):
        return await self.check(0, option="gt")

    async def wait(self, timeout: Optional[float] = None):
        """
        Wait until the lock is released.
        """
        return await self.wait_for(0, option="eq", timeout=timeout)
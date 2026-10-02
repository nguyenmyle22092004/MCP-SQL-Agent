from pydantic import BaseModel, Field


class MemoryRecord(BaseModel):
    id: int | None = None
    user_id: str
    memory_type: str
    content: str
    metadata: dict = Field(default_factory=dict)


class ShortTermMessage(BaseModel):
    role: str
    content: str

from pydantic import BaseModel

class ChatRequest(BaseModel):
    prompt: str
    temperature: float = 0.7

class ChatResponse(BaseModel):
    response: str

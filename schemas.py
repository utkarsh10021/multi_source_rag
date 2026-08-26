from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    question: str = Field(min_length=1)

class CrawlRequest(BaseModel):
    url: str = Field(min_length=1)

class IngestResponse(BaseModel):
    documents: int
    chunks: int
    message: str

class AskSource(BaseModel):
    number: int
    title: str
    source: str
    source_type: str
    score: float
    content: str

class AskResult(BaseModel):
    answer: str
    rewritten_query: str
    sources: list[AskSource]

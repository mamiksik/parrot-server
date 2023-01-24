from pydantic import BaseModel


class InputPayload(BaseModel):
    inputs: str


class SummarizePayload(InputPayload):
    min_length: int = 2
    max_length: int = 100
    num_beams: int = 7
    prediction_count: int = 5


class Prediction(BaseModel):
    score: int
    token_str: str

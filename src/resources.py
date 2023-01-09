from pydantic import BaseModel


class InputPayload(BaseModel):
    inputs: str


class SummarizePayload(InputPayload):
    lang: str
    min_length: int = 5
    max_length: int = 100
    num_beams: int = 7
    prediction_count: int = 5


class Prediction(BaseModel):
    score: str
    token_str: str

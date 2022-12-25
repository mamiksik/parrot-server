from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline, Pipeline

app = FastAPI()
shared_resources = {}


@app.on_event("startup")
async def startup_event():
    tokenizer = AutoTokenizer.from_pretrained("mamiksik/CommitPredictor")
    model = AutoModelForMaskedLM.from_pretrained("mamiksik/CommitPredictor")
    shared_resources['pipe'] = pipeline("fill-mask", model=model, tokenizer=tokenizer)


class Payload(BaseModel):
    inputs: str


class Prediction(BaseModel):
    score: str
    token_str: str


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def say_hello(payload: Payload) -> list[Prediction]:
    pipe: Pipeline = shared_resources['pipe']
    result = pipe.predict(payload.inputs)
    return [Prediction(score=prediction["score"], token_str=prediction["token_str"]) for prediction in result]

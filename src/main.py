from functools import lru_cache

import torch
from fastapi import FastAPI
from transformers import pipeline

from src.resources import Prediction, SummarizePayload, InputPayload
from src.utils import model_tokenizer, parse_git_diff

app = FastAPI()


@app.get("/")
async def status():
    return {"status": "ok"}


@app.post("/fill-token")
async def predict_compatibility(payload: InputPayload) -> list[Prediction]:
    tokenizer, model = model_tokenizer(
        "mamiksik/CommitPredictor", "mamiksik/CommitPredictor", "roberta"
    )
    pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer)

    result = pipe.predict(payload.inputs)
    return [
        Prediction(score=prediction["score"], token_str=prediction["token_str"])
        for prediction in result
    ]


@app.post("/summarize")
async def summarize(payload: SummarizePayload) -> list[Prediction]:
    tokenizer, model = model_tokenizer(
        "mamiksik/CommitPredictorT5PL", "mamiksik/CommitPredictorT5PL", "t5", model_revision='fb08d01'
    )

    # If input is empty
    if not payload.inputs:
        return []

    with torch.no_grad():
        input_ids = tokenizer(
            payload.inputs,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).input_ids

        outputs = model.generate(
            input_ids,
            max_length=payload.max_length,
            min_length=payload.min_length,
            num_beams=payload.num_beams,
            num_return_sequences=payload.prediction_count,
        )

    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [Prediction(score=-1, token_str=prediction) for prediction in result]


@app.post("/summarize-raw")
async def summarize_raw(payload: SummarizePayload) -> list[Prediction]:
    payload.inputs = parse_git_diff(payload.inputs)
    result = await summarize(payload)
    return result

from functools import lru_cache

import torch
from fastapi import FastAPI
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    pipeline,
    T5ForConditionalGeneration,
    RobertaTokenizer,
)

from src.resources import Prediction, SummarizePayload, InputPayload

app = FastAPI()
shared_resources = {}


@lru_cache()
def model_tokenizer(tokenizer_name, model_name, type):
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    if type == "t5":
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    elif type == "roberta":
        model = AutoModelForMaskedLM.from_pretrained(model_name)
    else:
        raise ValueError("Invalid model type")

    return tokenizer, model


@app.on_event("startup")
async def startup_event():
    tokenizer = AutoTokenizer.from_pretrained("mamiksik/CommitPredictor")
    model = AutoModelForMaskedLM.from_pretrained("mamiksik/CommitPredictor")
    shared_resources["pipe"] = pipeline("fill-mask", model=model, tokenizer=tokenizer)


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
        "mamiksik/CommitPredictorT5", "mamiksik/CommitPredictorT5", "t5"
    )

    with torch.no_grad():
        input_ids = tokenizer(
            f"Summarize {payload.lang}: {payload.inputs}",
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
    return [Prediction(score=1, token_str=prediction) for prediction in result]

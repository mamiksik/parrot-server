import re
from functools import lru_cache

from transformers import T5ForConditionalGeneration, RobertaTokenizer, AutoModelForMaskedLM, AutoTokenizer


@lru_cache()
def model_tokenizer(tokenizer_name, model_name, model_type, *, model_revision='main'):
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name, revision=model_revision)
    if model_type == "t5":
        model = T5ForConditionalGeneration.from_pretrained(model_name, revision=model_revision)
    elif model_type == "roberta":
        model = AutoModelForMaskedLM.from_pretrained(model_name, revision=model_revision)
    else:
        raise ValueError("Invalid model type")

    return tokenizer, model


def parse_git_diff(patch):
    accumulator = []
    lines = patch.splitlines()

    filename_before = None
    for line in lines:
        if line.startswith("index") or line.startswith("diff"):
            continue
        if line.startswith("---"):
            filename_before = line.split(" ", 1)[1][1:]
            continue

        if line.startswith("+++"):
            filename_after = line.split(" ", 1)[1][1:]

            if filename_before == filename_after:
                accumulator.append(f"<ide><path>{filename_before}")
            else:
                accumulator.append(f"<add><path>{filename_after}")
                accumulator.append(f"<del><path>{filename_before}")
            continue

        line = re.sub("@@[^@@]*@@", "", line)
        if len(line) == 0:
            continue

        if line[0] == "+":
            line = line.replace("+", "<add>", 1)
        elif line[0] == "-":
            line = line.replace("-", "<del>", 1)
        else:
            line = f"<ide>{line}"

        accumulator.append(line)

    return '\n'.join(accumulator)
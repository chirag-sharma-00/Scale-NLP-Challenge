import sys
import numpy as np
import re
from typing import Tuple
import torch
import transformers
import train

MAX_SEQUENCE_LENGTH = 29
TRAIN_URL = "https://scale-static-assets.s3-us-west-2.amazonaws.com/ml-interview/expand/train.txt"


def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """ A helper functions that loads the file into a tuple of strings

    :param file_path: path to the data file
    :return factors: (LHS) inputs to the model
            expansions: (RHS) group truth
    """
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])
    return factors, expansions


def score(true_expansion: str, pred_expansion: str) -> int:
    """ the scoring function - this is how the model will be evaluated

    :param true_expansion: group truth string
    :param pred_expansion: predicted string
    :return:
    """
    return int(true_expansion == pred_expansion)


# --------- START OF IMPLEMENT THIS --------- #

def predict(factors: str, model: torch.nn.Module, tokenizer: transformers.PreTrainedTokenizer):
    match = re.search("[a-z](?!\(*[a-z]+)", factors)
    variable = factors[match.start(0):match.end(0)]
    data = re.sub("[a-z](?!\(*[a-z]+)", "x", factors)
    inputs = tokenizer(data, text_pair=None,
                       return_tensors="pt", padding="max_length",
                       add_special_tokens=False,
                       return_token_type_ids=False)
    pred = tokenizer.decode(model.generate(**inputs).squeeze(), skip_special_tokens=True)
    pred = re.sub("[a-z](?!\(*[a-z]+)", variable, pred).replace(' ', '')
    return pred


# --------- END OF IMPLEMENT THIS --------- #


def main(filepath: str):
    factors, expansions = load_file(filepath)
    model, tokenizer = train.create_model(tokens_file="tokens.txt", num_layers=1)
    model.load_state_dict(torch.load("encoder_decoder_lr=0.0005_epochs=6_batch_size=1024_seed=12321.pt", map_location=torch.device('cpu')))
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters:", pytorch_total_params)
    pred = [predict(f, model, tokenizer) for f in factors]
    scores = [score(te, pe) for te, pe in zip(expansions, pred)]
    print(np.mean(scores))


if __name__ == "__main__":
    main("valid.txt")
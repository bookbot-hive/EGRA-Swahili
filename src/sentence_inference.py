#!/usr/bin/env python3
import argparse
from typing import Dict, List
import json
import os
import re

import torch
from datasets import Audio, Dataset, load_dataset
import evaluate
from transformers import AutoFeatureExtractor, pipeline


def log_results(result: Dataset, args: Dict[str, str]):
    """DO NOT CHANGE. This function computes and logs the result metrics."""
    model_id = args.model.split("/")[-1]
    dataset_id = "_".join(args.dataset.split("/") + [args.split])

    # load metric
    wer = evaluate.load("wer")
    cer = evaluate.load("cer")

    # compute metrics
    wer_result = wer.compute(references=result["target"], predictions=result["prediction"])
    cer_result = cer.compute(references=result["target"], predictions=result["prediction"])

    # print & log results
    result_str = f"PER: {wer_result}\n" f"CER: {cer_result}"
    print(result_str)

    logging_dir = f"{args.logging_dir}/{model_id}"
    os.makedirs(logging_dir, exist_ok=True)

    with open(f"{logging_dir}/metrics_{dataset_id}.txt", "w") as f:
        f.write(result_str)

    with open(f"{logging_dir}/log_{dataset_id}.json", "w") as f:
        data = [{"prediction": p, "target": t} for p, t in zip(result["prediction"], result["target"])]
        json.dump(data, f, indent=2, ensure_ascii=False)


def timestamps_to_transcript(timestamps) -> str:
    """Formats `AutomaticSpeechRecognitionPipeline`'s timestamp outputs to a transcript string."""
    return " ".join([o["text"] if o["text"] != " " else "|" for o in timestamps["chunks"]]).strip("|").strip()


def normalize_text(phonemes: List[str]) -> str:
    return " | ".join(phonemes)


def main(args):
    # load dataset
    dataset = load_dataset(args.dataset, split=args.split)

    # load processor
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model)
    sampling_rate = feature_extractor.sampling_rate

    # resample audio
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

    # load eval pipeline
    device = 0 if torch.cuda.is_available() else -1
    asr = pipeline("automatic-speech-recognition", model=args.model, device=device)

    chars_to_ignore_regex = f'[{"".join(args.chars_to_ignore)}]' if args.chars_to_ignore is not None else None

    # map function to decode audio
    def map_to_pred(batch):
        prediction = asr(
            batch["audio"]["array"],
            return_timestamps="char",
        )

        batch["prediction"] = timestamps_to_transcript(prediction)
        batch["target"] = re.sub(chars_to_ignore_regex, "", normalize_text(batch[args.text_column_name]))
        return batch

    # run inference on all examples
    result = dataset.map(map_to_pred, remove_columns=dataset.column_names)

    log_results(result, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model identifier. Should be loadable with 🤗 Transformers",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name to evaluate the `model_id`. Should be loadable with 🤗 Datasets",
    )
    parser.add_argument("--split", type=str, required=True, help="Split of the dataset. *E.g.* `'test'`")
    parser.add_argument(
        "--text_column_name",
        type=str,
        default="text",
        help="The name of the dataset column containing the text data. Defaults to 'text'",
    )
    parser.add_argument(
        "--chars_to_ignore",
        nargs="+",
        default=None,
        help="A list of characters to remove from the transcripts.",
    )
    parser.add_argument("--logging_dir", type=str, default="results", help="Path to logging directory.")
    args = parser.parse_args()

    main(args)

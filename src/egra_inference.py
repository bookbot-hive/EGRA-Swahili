from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import json

from tqdm.auto import tqdm
from transformers import pipeline
from datasets import load_dataset, Audio
import numpy as np
import Levenshtein

"""
Usage:

python src/egra_inference.py \
    --model_name "bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa" \
    --dataset_name "bookbot/bookbot_swahili_egra" \
    --split_name "test" \
    --subtask "word"
"""


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--split_name", type=str, default="test")
    parser.add_argument("--subtask", type=str, choices=["syllable", "word", "letter", "phoneme"], required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--audio_column_name", type=str, default="audio")
    parser.add_argument("--label_column_name", type=str, default="phonemes")
    parser.add_argument("--min_length_s", type=float, default=1.0)
    parser.add_argument("--output_dir", type=Path, default=Path("results"))
    parser.add_argument("--use_substitution_pairs", action="store_true")
    return parser.parse_args()


def phoneme_error_rate(labels, predictions, substitution_pairs=[]):
    def get_errors(label, prediction, substitution_pairs=[]):
        _label, _pred = label.split(), prediction.split()
        n = len(_label)
        s, d, i = 0, 0, 0

        for tag, i1, i2, j1, j2 in Levenshtein.opcodes(_label, _pred):
            l = " ".join(_label[i1:i2])
            p = " ".join(_pred[j1:j2])

            if tag == "replace" and not ((l, p) in substitution_pairs or (p, l) in substitution_pairs):
                s += 1
            elif tag == "delete":
                d += 1
            elif tag == "insert":
                i += 1

        return s, d, i, n

    all_errors = [get_errors(label, prediction, substitution_pairs) for label, prediction in zip(labels, predictions)]
    s, d, i, n = map(sum, zip(*all_errors))
    return (s + d + i) / n


def calculate_error_stats(labels, predictions, substitution_pairs=[]):
    substitutions, insertions, deletions = Counter(), Counter(), Counter()
    errors = Counter([f"{l} -> {p}" for (l, p) in zip(labels, predictions) if l != p])

    for label, pred in zip(labels, predictions):
        _label, _pred = label.split(), pred.split()

        for tag, i1, i2, j1, j2 in Levenshtein.opcodes(_label, _pred):
            l = " ".join(_label[i1:i2])
            p = " ".join(_pred[j1:j2])

            if tag == "replace" and not ((l, p) in substitution_pairs or (p, l) in substitution_pairs):
                substitutions[f"{l} -> {p}"] += 1
            elif tag == "insert":
                insertions[p] += 1
            elif tag == "delete":
                deletions[l] += 1

    return {
        "substitutions": dict(substitutions.most_common()),
        "insertions": dict(insertions.most_common()),
        "deletions": dict(deletions.most_common()),
        "errors": dict(errors.most_common()),
    }


def main(args):
    args.output_dir.mkdir(exist_ok=True, parents=True)

    transcriber = pipeline(
        "automatic-speech-recognition",
        model=args.model_name,
        device=args.device,
        return_timestamps="char",
    )
    sampling_rate = transcriber.feature_extractor.sampling_rate

    dataset = load_dataset(args.dataset_name, split=args.split_name)
    dataset = dataset.cast_column(args.audio_column_name, Audio(sampling_rate=sampling_rate))
    dataset = dataset.filter(lambda x: x == args.subtask, input_columns="subtask")

    def transcribe(datum):
        input_samples = datum[args.audio_column_name]["array"]

        min_length = int(args.min_length_s * sampling_rate)
        if len(input_samples) < min_length:  # right-pad short audio files
            input_samples = np.pad(input_samples, (0, min_length - len(input_samples)), mode="constant")

        outputs = transcriber(input_samples)
        predictions = [o["text"] for o in outputs["chunks"] if len(o["text"].strip()) > 0]
        return " ".join(predictions).strip()

    predictions, labels = [], []
    for datum in tqdm(dataset):
        prediction = transcribe(datum)
        label = datum[args.label_column_name]
        predictions.append(prediction)
        labels.append(label)

    if args.use_substitution_pairs:
        substitution_pairs = [
            ("ð", "v"),
            ("ɣ", "ɠ"),
            ("ⁿz", "z"),
            ("x", "h"),
            ("w", "u"),
            ("ʃ", "s"),
            ("θ", "ð"),
            ("ⁿɗ͡ʒ", "ʄ"),
            ("ɗ", "ð"),
            ("θ", "t"),
        ]
    else:
        substitution_pairs = []

    per = phoneme_error_rate(labels, predictions, substitution_pairs)

    results = {
        "model": args.model_name,
        "dataset": args.dataset_name,
        "subtask": args.subtask,
        "per": per,
        "substitution_pairs": substitution_pairs,
        **calculate_error_stats(labels, predictions, substitution_pairs),
    }

    model_name = args.model_name.split("/")[-1]
    dataset_name = args.dataset_name.split("/")[-1]

    with open(f"{args.output_dir}/results_{model_name}_{dataset_name}_{args.subtask}.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)

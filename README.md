# EGRA-Swahili

To run EGRA tasks, you need to specify the model, dataset, and split name. The `subtask` argument can be one of `syllable`, `word`, `letter`, `pseudo_word`, `sentence`, or `phoneme`. At the moment, we support wav2vec2-CTC models that are integrated with the Hugging Face transformers library.

## Datasets on 🤗 Hub

The dataset is available on the Hugging Face Datasets Hub (with private access):

- [Swahili EGRA Kids](https://huggingface.co/datasets/bookbot/bookbot_swahili_egra_kids)

```sh
model=bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-alphabets-phonemes-bookbot
split=test

for subtask in syllable letter phoneme pseudo_word word sentence; do
    python src/egra_inference.py \
        --model_name $model \
        --dataset_name bookbot/bookbot_swahili_egra_kids \
        --split_name $split \
        --subtask $subtask \
        --use_substitution_pairs
done
```

## Local Datasets

To run the code locally, you need to prepare a directory with the following structure:

<details>
  <summary>Directory Structure</summary>
  
```
bookbot_swahili_egra_kids/
├── letter
│   ├── a.txt
│   ├── a.wav
│   ├── ...
├── phoneme
│   ├── ɑ.txt
│   ├── ɑ.wav
│   ├── ...
├── pseudo_word
|   ├── adoseti.txt
|   ├── adoseti.wav
|   ├── ...
├── sentence
│   ├── sentence1.txt
│   ├── sentence1.wav
│   ├── ...
├── syllable
│   ├── kwa.txt
│   ├── kwa.wav
│   ├── ...
└── word
    ├── mbali.txt
    ├── mbali.wav
    ├── ...
```
</details>

where each subdirectory is a subtask, and each audio `.wav` file has a corresponding phoneme transcript `.txt` file. The audio files should be in the WAV format, and will be resampled to 16kHz. The transcripts should contain the phoneme labels for the corresponding audio files. The phoneme labels should be in the format of a single line with space-separated phonemes (e.g. `n i p ɑ k ɑ m k u ɓ w ɑ`). We use phoneme vocabulary from the [gruut](https://github.com/rhasspy/gruut) phonemizer. For a full list of the vocab, please see [here](https://huggingface.co/bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-alphabets-phonemes-bookbot/blob/main/vocab.json).

Then to run the code locally, you can use the following command:

```sh
model=bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-alphabets-phonemes-bookbot

for subtask in syllable letter phoneme pseudo_word word sentence; do
    python src/egra_inference.py \
        --model_name $model \
        --dataset_dir /path/to/bookbot_swahili_egra_kids/ \
        --subtask $subtask \
        --use_substitution_pairs
done
```

## Results

The following table shows the performance of the model on the test sets:

Model: [`bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-alphabets-phonemes-bookbot`](https://huggingface.co/bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-alphabets-phonemes-bookbot)

| Subtask     | PER (%) |
| ----------- | ------: |
| Sentence    |    0.41 |
| Word        |    0.57 |
| Pseudo Word |    4.85 |
| Syllable    |    4.58 |
| Letter      |    2.38 |
| Phoneme     |    9.09 |
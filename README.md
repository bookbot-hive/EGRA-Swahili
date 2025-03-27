# EGRA-Swahili

## Letter-level Tasks

To run letter-level tasks, you need to specify the model, dataset, and split name. The `subtask` argument can be one of `syllable`, `word`, `letter`, `pseudo_word` or `phoneme`.
At the moment, we support wav2vec2-CTC models that are integrated with the Hugging Face transformers library.

```sh
model=bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-alphabets-phonemes-bookbot
split=test

for subtask in syllable letter phoneme pseudo_word; do
    python src/egra_inference.py \
        --model_name $model \
        --dataset_name bookbot/bookbot_swahili_egra_althaf \
        --split_name $split \
        --subtask $subtask \
        --use_substitution_pairs
done
```

### Dataset

We provide letter-level tasks which include syllable, word, letter, phoneme, and pseudo-word subtasks. The datasets are available on the Hugging Face Datasets Hub:

- [Swahili EGRA Althaf (Kids)](https://huggingface.co/datasets/bookbot/bookbot_swahili_egra_althaf)

> ‼️ Note: Swahili EGRA Althaf currently do not support the single-word task as a private evaluation was conducted.

## Sentence-level Task

Similarly we provide a script to run the sentence-level task. The script requires the model, dataset, split, and the column name of the text to be transcribed.

```sh
model=bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-alphabets-phonemes-bookbot

for dataset in bookbot/common_voice_16_1_sw bookbot/fleurs_sw; do
    python src/sentence_inference.py \
        --model $model \
        --dataset $dataset \
        --split test \
        --text_column_name phonemes_ipa \
        --chars_to_ignore , ? . ! - \; \: \" “ % ‘ ” �
done
```

### Datasets

We provide pre-phonemized, sentence-level task datasets on the Hugging Face Datasets Hub:

- [Common Voice Swahili](https://huggingface.co/datasets/bookbot/common_voice_16_1_sw)
- [FLEURS Swahili](https://huggingface.co/datasets/bookbot/fleurs_sw)

> ‼️ Note: Likewise, the datasets shared above are publicly available. The datasets used for the private evaluation (word and sentence tasks) are not shared publicly.

## Results

The following table shows the performance of the model on the test sets:

Model: [`bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-alphabets-phonemes-bookbot`](https://huggingface.co/bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-alphabets-phonemes-bookbot)

| Dataset     | Subtask     | PER (%) |
| ----------- | ----------- | ------: |
| EGRA Althaf | Sentence    |    0.50 |
| EGRA Althaf | Word        |    1.10 |
| EGRA Althaf | Pseudo Word |    4.85 |
| EGRA Althaf | Syllable    |    4.58 |
| EGRA Althaf | Letter      |    2.38 |
| EGRA Althaf | Phoneme     |    9.09 |
# EGRA-Swahili

## EGRA

### Inference

To run the EGRA inference script, you need to specify the model, dataset, and split name. The `subtask` argument can be one of `syllable`, `word`, or `letter`.
At the moment, we support wav2vec2-CTC models that are integrated with the Hugging Face transformers library.

```sh
model=bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-easyswahili
dataset=bookbot/bookbot_swahili_egra
split=test

for subtask in syllable word letter phoneme; do
    python src/egra_inference.py \
        --model_name $model \
        --dataset_name $dataset \
        --split_name $split \
        --subtask $subtask \
        --use_substitution_pairs
done
```

### EGRA Dataset

The EGRA Swahili dataset is a collection of audio recordings and transcriptions of Swahili text. The dataset is split into subtasks: syllable, word, phoneme, and letter. The dataset is available on the Hugging Face Datasets Hub [here](https://huggingface.co/datasets/bookbot/bookbot_swahili_egra/). The dataset contains the following columns:

| audio                                           | transcript | phonemes         | subtask  |
| ----------------------------------------------- | ---------- | ---------------- | -------- |
| syllable/sw-TZ-Victoria_syllable_1150_0_nte.wav | nte        | n t ɛ            | syllable |
| syllable/sw-TZ-Victoria_syllable_1150_0_sa.wav  | sa         | s ɑ              | syllable |
| word/sw-TZ-Victoria_001158_upinzani.wav         | upinzani   | u p i ⁿz ɑ n i   | word     |
| word/sw-TZ-Victoria_001169_kiburunzi.wav        | kiburunzi  | k i ɓ u ɾ u ⁿz i | word     |
| letter/a.wav                                    | a          | ɑ                | letter   |
| letter/ba.wav                                   | ba         | ɓ ɑ              | letter   |
| letter/cha.wav                                  | cha        | t͡ʃ ɑ             | letter   |
| phoneme/ð.wav                                   | ð          | ð                | phoneme  |
| phoneme/ɛ.wav                                   | ɛ          | ɛ                | phoneme  |

### Results

The following table shows the performance of the model on the test set of the EGRA Swahili dataset.

Model: [`bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-easyswahili`](https://huggingface.co/bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-easyswahili)

| Subtask  | PER (%) |
| -------- | ------: |
| Syllable |    4.25 |
| Word     |    3.13 |
| Letter   |    8.62 |

## Sentence-level Task

### Inference

```sh
MODEL="bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-easyswahili"

python src/inference.py \
    --model=$MODEL \
    --dataset="bookbot/common_voice_16_1_sw" \
    --split="test" \
    --text_column_name="phonemes_ipa" \
    --chars_to_ignore , ? . ! - \; \: \" “ % ‘ ” �

python src/inference.py \
    --model=$MODEL \
    --dataset="bookbot/fleurs_sw" \
    --split="test" \
    --text_column_name="phonemes_ipa" \
    --chars_to_ignore , ? . ! - \; \: \" “ % ‘ ” �
```

### Datasets

We provide pre-phonemized, sentence-level task datasets on the Hugging Face Datasets Hub:

- [Common Voice Swahili](https://huggingface.co/datasets/bookbot/common_voice_16_1_sw)
- [FLEURS Swahili](https://huggingface.co/datasets/bookbot/fleurs_sw)

### Results

Model: [`bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-easyswahili`](https://huggingface.co/bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-easyswahili)

| Dataset      | PER (%) |
| ------------ | ------: |
| Common Voice |    6.52 |
| FLEURS       |    4.85 |
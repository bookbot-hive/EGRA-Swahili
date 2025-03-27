# EGRA-Swahili

To run EGRA tasks, you need to specify the model, dataset, and split name. The `subtask` argument can be one of `syllable`, `word`, `letter`, `pseudo_word`, `sentence`, or `phoneme`. At the moment, we support wav2vec2-CTC models that are integrated with the Hugging Face transformers library.

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

## Dataset

The dataset is available on the Hugging Face Datasets Hub (with private access):

- [Swahili EGRA Kids](https://huggingface.co/datasets/bookbot/bookbot_swahili_egra_kids)

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
# EGRA-Swahili

## Usage

```sh
python src/egra_inference.py \
    --model_name "bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa" \
    --dataset_name "bookbot/bookbot_swahili_egra" \
    --split_name "test" \
    --subtask "syllable" \
    --use_substitution_pairs

python src/egra_inference.py \
    --model_name "bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa" \
    --dataset_name "bookbot/bookbot_swahili_egra" \
    --split_name "test" \
    --subtask "word" \
    --use_substitution_pairs
```

## Dataset Format

| audio                                           | transcript | phonemes         | subtask  |
| ----------------------------------------------- | ---------- | ---------------- | -------- |
| syllable/sw-TZ-Victoria_syllable_1150_0_a.wav   | a          | ɑ                | syllable |
| syllable/sw-TZ-Victoria_syllable_1150_0_nte.wav | nte        | n t ɛ            | syllable |
| syllable/sw-TZ-Victoria_syllable_1150_0_sa.wav  | sa         | s ɑ              | syllable |
| syllable/sw-TZ-Victoria_syllable_1150_1_kwa.wav | kwa        | k w ɑ            | syllable |
| syllable/sw-TZ-Victoria_syllable_1150_2_a.wav   | a          | ɑ                | syllable |
| word/sw-TZ-Victoria_001158_upinzani.wav         | upinzani   | u p i ⁿz ɑ n i   | word     |
| word/sw-TZ-Victoria_001169_kiburunzi.wav        | kiburunzi  | k i ɓ u ɾ u ⁿz i | word     |
| word/sw-TZ-Victoria_001169_mdanzi.wav           | mdanzi     | m ɗ ɑ ⁿz i       | word     |
| word/sw-TZ-Victoria_001169_sunzika.wav          | sunzika    | s u ⁿz i k ɑ     | word     |
| word/sw-TZ-Victoria_001169_uunzi.wav            | uunzi      | u u ⁿz i         | word     |
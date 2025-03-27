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
model=bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-alphabets-phonemes
dataset=bookbot/bookbot_swahili_egra
split=test

for subtask in pseudo_word; do
    python src/egra_inference.py \
        --model_name $model \
        --dataset_name $dataset \
        --split_name $split \
        --subtask $subtask \
        --use_substitution_pairs
done
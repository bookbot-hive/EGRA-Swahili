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

python src/egra_inference.py \
    --model_name "bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa" \
    --dataset_name "bookbot/bookbot_swahili_egra" \
    --split_name "test" \
    --subtask "letter" \
    --use_substitution_pairs
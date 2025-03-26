MODEL="bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-alphabets-phonemes"

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
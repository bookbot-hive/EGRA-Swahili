model="bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-alphabets-phonemes"

for dataset in "bookbot/common_voice_16_1_sw" "bookbot/fleurs_sw"; do
    python src/sentence_inference.py \
        --model=$model \
        --dataset=$dataset \
        --split="test" \
        --text_column_name="phonemes_ipa" \
        --chars_to_ignore , ? . ! - \; \: \" “ % ‘ ” �
done
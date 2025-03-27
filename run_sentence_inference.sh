model=bookbot/wav2vec2-xls-r-300m-swahili-cv-fleurs-alffa-alphabets-phonemes-bookbot

for dataset in bookbot/common_voice_16_1_sw bookbot/fleurs_sw; do
    python src/sentence_inference.py \
        --model $model \
        --dataset $dataset \
        --split test \
        --text_column_name phonemes_ipa \
        --chars_to_ignore , ? . ! - \; \: \" “ % ‘ ” �
done

python src/sentence_inference.py \
    --model $model \
    --dataset "bookbot/bookbot_sw_v4" \
    --split test \
    --text_column_name phonemes \
    --chars_to_ignore , ? . ! - \; \: \" “ % ‘ ” �
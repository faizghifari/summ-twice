# pip install methodtools py-rouge pyrouge nltk transformers bertopic datasets
python -c 'import nltk; nltk.download("punkt"); nltk.download("stopwords")'

python main.py \
    --data_path ./data/qmsum/test.jsonl \
    --output_path ./output/test-bart_large-beam-linear.json \
    --model_name_or_path ./models/beam/QMSum-BART-large-beam \
    --max_model_len 1024 \
    --max_seg_tgt_len 50 \
    --max_tgt_len 256 \
    --min_tgt_len 20 \
    --segmenter_type linear \
    --doc_len_threshold 5

# python main.py \
#     --data_path ./data/qmsum/test.jsonl \
#     --output_path ./output/test-pegasus_large-beam-linear.json \
#     --model_name_or_path ./models/beam/QMSum-PEGASUS-large-beam \
#     --max_model_len 1024 \
#     --max_seg_tgt_len 50 \
#     --max_tgt_len 256 \
#     --min_tgt_len 20 \
#     --segmenter_type linear \
#     --doc_len_threshold 5

# python main.py \
#     --data_path ./data/qmsum/test.jsonl \
#     --output_path ./output/test-led_base-beam-linear.json \
#     --model_name_or_path ./models/beam/QMSum-LED-base-beam \
#     --max_model_len 16384 \
#     --max_seg_tgt_len 50 \
#     --max_tgt_len 256 \
#     --min_tgt_len 20 \
#     --segmenter_type linear \
#     --doc_len_threshold 5

# python main.py \
#     --data_path ./data/qmsum/test.jsonl \
#     --output_path ./output/test-bart_large-contrastive-linear.json \
#     --model_name_or_path ./models/contrastive/QMSum-BART-large \
#     --max_model_len 1024 \
#     --max_seg_tgt_len 50 \
#     --max_tgt_len 256 \
#     --min_tgt_len 20 \
#     --segmenter_type linear \
#     --doc_len_threshold 5

# python main.py \
#     --data_path ./data/qmsum/test.jsonl \
#     --output_path ./output/test-pegasus_large-contrastive-linear.json \
#     --model_name_or_path ./models/contrastive/QMSum-PEGASUS-large \
#     --max_model_len 1024 \
#     --max_seg_tgt_len 50 \
#     --max_tgt_len 256 \
#     --min_tgt_len 20 \
#     --segmenter_type linear \
#     --doc_len_threshold 5

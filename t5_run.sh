# pip install methodtools py-rouge pyrouge nltk transformers bertopic datasets
# python -c 'import nltk; nltk.download("punkt"); nltk.download("stopwords")'

python main.py \
    --data_path ./data/qmsum/test.jsonl \
    --output_path ./output/test-t5_3b-beam-bertopic.json \
    --model_name_or_path ./models/beam/QMSum-T5-3B-beam \
    --max_model_len 512 \
    --max_seg_tgt_len 128 \
    --max_tgt_len 256 \
    --min_tgt_len 20 \
    --segmenter_type bertopic \
    --doc_len_threshold 5

python main.py \
    --data_path ./data/qmsum/test.jsonl \
    --output_path ./output/test-t5_large-beam-bertopic.json \
    --model_name_or_path ./models/beam/QMSum-T5-large-beam \
    --max_model_len 512 \
    --max_seg_tgt_len 128 \
    --max_tgt_len 256 \
    --min_tgt_len 20 \
    --segmenter_type bertopic \
    --doc_len_threshold 5

python main.py \
    --data_path ./data/qmsum/test.jsonl \
    --output_path ./output/test-t5_v1_1_xl-beam-bertopic.json \
    --model_name_or_path ./models/beam/QMSum-T5_v1_1-xl-beam \
    --max_model_len 512 \
    --max_seg_tgt_len 128 \
    --max_tgt_len 256 \
    --min_tgt_len 20 \
    --segmenter_type bertopic \
    --doc_len_threshold 5

python main.py \
    --data_path ./data/qmsum/test.jsonl \
    --output_path ./output/test-t5_v1_1_large-beam-bertopic.json \
    --model_name_or_path ./models/beam/QMSum-T5_v1_1-large-beam \
    --max_model_len 512 \
    --max_seg_tgt_len 128 \
    --max_tgt_len 256 \
    --min_tgt_len 20 \
    --segmenter_type bertopic \
    --doc_len_threshold 5

python main.py \
    --data_path ./data/qmsum/test.jsonl \
    --output_path ./output/test-flant5_large-beam-bertopic.json \
    --model_name_or_path ./models/beam/QMSum-FLAN_T5-large-beam \
    --max_model_len 512 \
    --max_seg_tgt_len 128 \
    --max_tgt_len 256 \
    --min_tgt_len 20 \
    --segmenter_type bertopic \
    --doc_len_threshold 5

python main.py \
    --data_path ./data/qmsum/test.jsonl \
    --output_path ./output/test-t5_large-contrastive-bertopic.json \
    --model_name_or_path ./models/contrastive/QMSum-T5-large \
    --max_model_len 512 \
    --max_seg_tgt_len 128 \
    --max_tgt_len 256 \
    --min_tgt_len 20 \
    --segmenter_type bertopic \
    --doc_len_threshold 5

python main.py \
    --data_path ./data/qmsum/test.jsonl \
    --output_path ./output/test-t5_v1_1_large-contrastive-bertopic.json \
    --model_name_or_path ./models/contrastive/QMSum-T5_v1_1-large \
    --max_model_len 512 \
    --max_seg_tgt_len 128 \
    --max_tgt_len 256 \
    --min_tgt_len 20 \
    --segmenter_type bertopic \
    --doc_len_threshold 5

python main.py \
    --data_path ./data/qmsum/test.jsonl \
    --output_path ./output/test-flant5_large-contrastive-bertopic.json \
    --model_name_or_path ./models/contrastive/QMSum-FLAN_T5-large \
    --max_model_len 512 \
    --max_seg_tgt_len 128 \
    --max_tgt_len 256 \
    --min_tgt_len 20 \
    --segmenter_type bertopic \
    --doc_len_threshold 5

import os
import sys
import json
import logging

from tqdm import tqdm
from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, pipeline

from cls_segment.clustering import SegmentClusterProcessor

SPLITS = ['train', 'val', 'test']
MAX_MODEL_LEN = 512
MIN_TARGET_LEN = 5
MAX_TARGET_LEN = 20
MAX_TARGET_SENT = 100
UTTER_LEN_THRESHOLD = 6
N_CLUSTER_QUERY = 2
N_BEAMS = 6
MODEL_ID = "google/flan-t5-base"

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

cls_segmenter = SegmentClusterProcessor(
    max_model_len=MAX_MODEL_LEN,
    max_target_sent=MAX_TARGET_SENT,
    utter_len_threshold=UTTER_LEN_THRESHOLD,
    n_cluster_query=N_CLUSTER_QUERY
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=1)

# dataset = DatasetDict()
for split in SPLITS:
    # seg_src = []
    # seg_tgt = []
    new_dataset = []
    with open(f'./data/{split}.jsonl') as f:
        for line in tqdm(f):
            utter = []
            utter_speaker = []
            data = json.loads(line)
            for turn in data['meeting_transcripts']:
                utter.append(turn['content'])
                utter_speaker.append(turn['speaker'] + ' : ' + turn['content'])
            cluster_data, total_segments = cls_segmenter.cluster_utter(utter, utter_speaker)
            max_tgt_len = MAX_MODEL_LEN // total_segments
            if max_tgt_len <= MAX_TARGET_LEN:
                max_tgt_len = 20
            combined_summ = []
            for segments in cluster_data:
                # topic_summ = []
                for s in segments:
                    query = "summarize: " + s
                    if len(query.split()) >= MIN_TARGET_LEN:
                        if len(query.split()) < max_tgt_len:
                            result = summarizer(query, min_length=MIN_TARGET_LEN, max_length=len(query.split()))
                        else:
                            result = summarizer(query, min_length=MIN_TARGET_LEN, max_length=max_tgt_len)
                        combined_summ.append(result[0]['summary_text'])
                # combined_summ.append(topic_summ)
            combined_summ = " ".join(combined_summ)
            for query_type in ['general_query_list', 'specific_query_list']:
                for q in data[query_type]:
                    new_dataset.append({
                        "src": f"{q['query']} \n {combined_summ}",
                        "tgt": q['answer']
                    })
    with open(f'./data/qmsum_combined_summ_{split}.jsonl', "w") as f:
        for i in range(len(new_dataset)):
            print(json.dumps(new_dataset[i]), file=f)
    #         seg_src.extend(src)
    #         seg_tgt.extend(tgt)
    # dataset[split] = Dataset.from_dict({
    #     'src': seg_src,
    #     'tgt': seg_tgt
    # })

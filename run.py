import json

from tqdm import tqdm
from cls_segment.clustering import SegmentClusterProcessor

SPLIT = 'val'
MAX_MODEL_LEN = 2048
MAX_TARGET_SENT = 100
UTTER_LEN_THRESHOLD = 6
N_CLUSTER_QUERY = 2

cls_segmenter = SegmentClusterProcessor(
    max_model_len=MAX_MODEL_LEN,
    max_target_sent=MAX_TARGET_SENT,
    utter_len_threshold=UTTER_LEN_THRESHOLD,
    n_cluster_query=N_CLUSTER_QUERY
)

cluster_data = []
with open(f'./qmsum_speaker_{SPLIT}.jsonl') as f:
    for line in tqdm(f):
        data = json.loads(line)
        src_speaker = data['src_speaker']
        source = data['src']
        query = data['query']
        target = data['tgt']
        utter = source.split('@SEP@')
        utter_speaker = src_speaker.split('@SEP@')
        cluster_data.extend(cls_segmenter.process_cluster_summary_utter(utter, utter_speaker, target, query))

print(f'Total cluster data from {SPLIT} split: {len(cluster_data)}')
with open('./qmsum_segment_data_utter_' + SPLIT + '.jsonl', 'w') as f:
    for i in range(len(cluster_data)):
        print(json.dumps(cluster_data[i]), file=f)
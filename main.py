import json
import torch
import argparse

from tqdm import tqdm

from topic.bertopic import BERTopicModel
from topic.linearseg import LinearSegmenter

from summarizer.evaluator import Evaluator
from summarizer.model import IterativeSummarizer

from utils.dataloader import DataLoader
from utils.tools import get_max_len_query

device = 0 if torch.cuda.is_available() else -1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--max_model_len', type=int, required=True)
    parser.add_argument('--max_seg_tgt_len', type=int, required=True)
    parser.add_argument('--max_tgt_len', type=int, required=True)
    parser.add_argument('--min_tgt_len', type=int, required=True)
    parser.add_argument('--segmenter_type', type=str, default='bertopic')
    parser.add_argument('--doc_len_threshold', type=int, default=5)
    parser.add_argument('--remove_noise', type=bool, default=True)
    return parser.parse_args()

def main(): 
    args = parse_args()
    # add fixed random seed
    data_loader = DataLoader(args.data_path)
    dataset = data_loader.load_data()

    max_len_query = get_max_len_query(dataset)
    max_length = args.max_model_len - max_len_query - args.max_seg_tgt_len - 7

    if args.segmenter_type == 'bertopic':
        segmenter = BERTopicModel(max_length, args.doc_len_threshold)
    elif args.segmenter_type == 'linear':
        segmenter = LinearSegmenter(max_length, args.doc_len_threshold)
    
    summarizer = IterativeSummarizer(
        args.model_name_or_path, 
        args.max_model_len, 
        args.max_seg_tgt_len,
        args.max_tgt_len,
        args.min_tgt_len,
        device=device)

    targets = []
    all_summaries = []
    for data in tqdm(dataset):
        segment_texts = segmenter.segmentize(data, remove_noise=args.remove_noise)
        for i in range(len(data['queries'])):
            all_summaries.append({
                "all_summaries": summarizer.summarize_texts(segment_texts, data['queries'][i])
            })
            targets.append(data['targets'][i])
    
    with open(args.output_path, "w") as f:
        for summaries in all_summaries:
            print(json.dumps(summaries), file=f)
    with open('./data/qmsum/target.txt', 'w') as f:
        f.writelines(targets)

if __name__ == '__main__':
    main()
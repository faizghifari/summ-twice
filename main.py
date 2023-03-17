import torch
import argparse

from topic.bertopic import BERTopicModel

from summarizer.model import IterativeSummarizer

from utils.dataloader import DataLoader
from utils.tools import get_max_len_query

device = 0 if torch.cuda.is_available() else -1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--max_model_len', type=int, required=True)
    parser.add_argument('--max_seg_tgt_len', type=int, required=True)
    parser.add_argument('--max_tgt_len', type=int, required=True)
    parser.add_argument('--min_tgt_len', type=int, required=True)
    parser.add_argument('--doc_len_threshold', type=int, default=5)
    return parser.parse_args()

def main():
    args = parse_args()
    data_loader = DataLoader(args.data_path)
    dataset = data_loader.load_data()

    max_len_query = get_max_len_query(dataset)
    max_length = args.max_model_len - max_len_query - args.max_seg_tgt_len

    bertopic_model = BERTopicModel(max_length, args.doc_len_threshold)
    summarizer = IterativeSummarizer(
        args.model_name_or_path, 
        args.max_model_len, 
        args.max_seg_tgt_len,
        args.max_tgt_len,
        args.min_tgt_len,
        device=device)

    for data in dataset:
        topic_texts = bertopic_model.get_topics_text(data['utter'], data['utter_speaker'])
        for query in data['queries']:
            all_summaries = summarizer.summarize_texts(topic_texts, query)
            final_summary = all_summaries[-1]
            
    # continue with the rest of your code

if __name__ == '__main__':
    main()
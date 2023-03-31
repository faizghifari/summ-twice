import json
import torch
import argparse

from tqdm import tqdm

from topic.bertopic import BERTopicModel
from topic.linearseg import LinearSegmenter

from summarizer.evaluator import Evaluator
from summarizer.model import Summarizer

from utils.dataloader import DataLoader
from utils.tools import get_max_len_query


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--target_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--results_path", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--max_model_len", type=int, required=True)
    parser.add_argument("--max_seg_tgt_len", type=int, required=True)
    parser.add_argument("--max_tgt_len", type=int, required=True)
    parser.add_argument("--min_tgt_len", type=int, required=True)
    parser.add_argument("--segmenter_type", type=str, default="bertopic")
    parser.add_argument("--summarizer_type", type=str, default="incremental")
    parser.add_argument("--doc_len_threshold", type=int, default=5)
    parser.add_argument("--remove_noise", type=bool, default=True)
    parser.add_argument("--num_beams", type=int, default=6)
    parser.add_argument("--penalty_alpha", type=int, default=0)
    parser.add_argument("--cuda_devices", type=int, default=torch.device('cpu'))
    parser.add_argument("--use_deepspeed", type=bool, default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    data_loader = DataLoader(args.data_path)
    dataset = data_loader.load_data()

    max_len_query = get_max_len_query(dataset)
    max_length = (
        args.max_model_len - max_len_query - args.max_seg_tgt_len - 7
    )

    if args.segmenter_type == "bertopic":
        segmenter = BERTopicModel(max_length, args.doc_len_threshold)
    elif args.segmenter_type == "linear":
        segmenter = LinearSegmenter(max_length, args.doc_len_threshold)

    summarizer = Summarizer(
        args.model_name_or_path,
        args.max_model_len,
        args.max_seg_tgt_len,
        args.max_tgt_len,
        args.min_tgt_len,
        num_beams=args.num_beams,
        penalty_alpha=args.penalty_alpha,
        use_deepspeed=args.use_deepspeed,
        device=args.cuda_devices,
    )

    all_summaries = []
    for data in tqdm(dataset):
        segment_texts = segmenter.segmentize(
            data, remove_noise=args.remove_noise
        )
        for i in tqdm(range(len(data["queries"]))):
            all_summaries.append(
                {
                    "all_summaries": summarizer.summarize_texts(
                        segment_texts, data["queries"][i], args.summarizer_type
                    )
                }
            )

    with open(args.output_path, "w") as f:
        for summaries in all_summaries:
            print(json.dumps(summaries), file=f)

    rouge_evaluator = Evaluator("rouge")
    bertscore_evaluator = Evaluator("bertscore")

    preds = [pred["all_summaries"][-1] for pred in all_summaries]
    targets = []
    with open(args.target_path) as f:
        for line in f:
            targets.append(line)

    results = {
        "ROUGE": rouge_evaluator.compute_metrics(preds, targets), 
        "BERTScore": bertscore_evaluator.compute_metrics(preds, targets)
    } 

    with open(args.results_path, "w") as f:
        print(json.dumps(results), file=f)


if __name__ == "__main__":
    main()

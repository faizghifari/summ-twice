import json
import argparse

from tqdm import tqdm

from summarizer.evaluator import Evaluator

from utils.dataloader import DataLoader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', type=str, required=True)
    parser.add_argument('--target_path', type=str, required=True)
    parser.add_argument('--results_path', type=str, required=True)
    return parser.parse_args()

def main(): 
    args = parse_args()

    evaluator = Evaluator('rouge')

    preds = []
    targets = []
    with open(args.pred_path) as f:
        for line in f:
            pred = json.loads(line)
            preds.append(pred['all_summaries'][-1])
    with open(args.target_path) as f:
        for line in f:
            targets.append(line)

    results = evaluator.compute_metrics(preds, targets)

    with open(args.results_path, 'w') as f:
        print(json.dumps(results), file=f)

if __name__ == '__main__':
    main()

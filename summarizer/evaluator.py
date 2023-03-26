import nltk
import numpy as np
from typing import List, Tuple
from datasets import load_metric


class Evaluator:
    def __init__(self, metric_name: str):
        self.metric = load_metric(metric_name)

    def postprocess_text(self, preds: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
        preds = [pred.strip().lower() for pred in preds]
        labels = [label.strip().lower() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(self, preds: List[str], targets: List[str]) -> dict:
        # Some simple post-processing
        preds, targets = self.postprocess_text(preds, targets)

        result = self.metric.compute(predictions=preds, references=targets, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        result = {k: round(v, 4) for k, v in result.items()}
        return result

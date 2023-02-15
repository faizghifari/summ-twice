import nltk
from rouge import Rouge

class TargetMatchSegmenter(object):
    def __init__(self, max_target_sent):
        self.evaluator = Rouge(metrics=['rouge-n', 'rouge-l'],
                               max_n=2,
                               limit_length=False,
                               apply_avg=True,
                               apply_best=True,
                               weight_factor=1.2)
        self.max_target_sent = max_target_sent

    def seg_based_on_rouge(self, src, tgt, name=None, verbose=False) -> (list, str):
        cur_new = ''
        best_score = 0
        best_sents = []
        seg = [(x, i) for i, x in enumerate(nltk.sent_tokenize(tgt))]
        total_len = len(seg)
        for i in range(min(self.max_target_sent, total_len)):
            scores = [(x, self.evaluator.get_scores(cur_new + ' ' + x, src), i) for x, i in seg]
            scores.sort(key=lambda x: -x[1]['rouge-1']['f'])
            cur_new += scores[0][0] + ' '
            seg = [x for x in seg if x[1] != scores[0][2]]
            cur_score = self.evaluator.get_scores(cur_new, src)['rouge-1']['f']
            if cur_score > best_score:
                best_score = cur_score
                best_sents.append(scores[0])
            else:
                break

        if verbose:
            print("id:", name, "input/output:", total_len, len(best_sents), "best:", best_score)
        best_string = list(set((x[0], x[2]) for x in best_sents))
        best_string.sort(key=lambda x: x[1])
        best_string = ' '.join([x[0] for x in best_string])

        return best_sents, best_string

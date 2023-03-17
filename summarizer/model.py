from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


class IterativeSummarizer:
    def __init__(self, model_name_or_path, max_model_len, max_seg_tgt_len, max_tgt_len, min_tgt_len, device=None):
        self.max_model_len = max_model_len
        self.max_seg_tgt_len = max_seg_tgt_len
        self.max_tgt_len = max_tgt_len
        self.min_tgt_len = min_tgt_len

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, return_dict=True)
        self.summarizer = pipeline("summarization", model=self.model, tokenizer=self.tokenizer, device=device)

    def summarize(self, text, max_length=50, min_length=5):
        summary = self.summarizer(text, max_length=max_length, min_length=min_length)
        summary = summary[0]['summary_text']
        
        return summary

    def summarize_texts(self, texts, query):
        all_summaries = []
        prev_summaries = []
        for i in range(len(texts)):
            if prev_summaries:
                summary_len = sum([len(summ.split()) for summ in prev_summaries])
                total_len = len(query.split()) + summary_len + len(texts[i].split())
                if total_len >= self.max_model_len:
                    prev_summary = self.summarize("\n".join(prev_summaries), max_length=self.max_seg_tgt_len, min_length=self.min_tgt_len)
                    prev_summaries = [prev_summary]

            input_text = f"{query}\n\n{"\n".join(prev_summaries)}\n{texts[i]}"
            if i != len(texts) - 1:
                summary = self.summarize(input_text, max_length=self.max_seg_tgt_len, min_length=self.min_tgt_len)
            else:
                summary = self.summarize(input_text, max_length=self.max_tgt_len, min_length=self.min_tgt_len)
            prev_summaries.append(summary)
            all_summaries.append(summary)
        
        return all_summaries

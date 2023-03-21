from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


class IterativeSummarizer:
    def __init__(self, model_name_or_path, max_model_len, max_seg_tgt_len, max_tgt_len, min_tgt_len, device=-1):
        self.max_model_len = max_model_len
        self.max_seg_tgt_len = max_seg_tgt_len
        self.max_tgt_len = max_tgt_len
        self.min_tgt_len = min_tgt_len

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, return_dict=True).to(device)
        self.summarizer = pipeline("summarization", model=self.model, tokenizer=self.tokenizer, device=device)

    def summarize(self, text, max_length=50, min_length=5):
        summary = self.summarizer(text, max_length=max_length, min_length=min_length, truncation=True)
        summary = summary[0]['summary_text']
        
        return summary

    def summarize_texts(self, texts, query):
        all_summaries = []
        for i in range(len(texts)):
            if all_summaries:
                summary_len = sum([len(summ.split()) for summ in all_summaries])
                total_len = len(query.split()) + summary_len + len(texts[i].split())
                context = "\n".join(all_summaries)
                if total_len >= self.max_model_len and len(context.split()) > self.max_seg_tgt_len:
                    context = self.summarize("\n".join(all_summaries), max_length=self.max_seg_tgt_len, min_length=self.min_tgt_len)
                input_text = f"Given this context and dialogue, {query}\n\nContext: {context}\n\nDialogue: {texts[i]}"
            else:
                input_text = f"Given this dialogue, {query}\n\nDialogue: {texts[i]}"
            
            if i != len(texts) - 1:
                if len(input_text.split()) < self.max_seg_tgt_len:
                    summary = self.summarize(input_text, max_length=len(input_text.split()), min_length=self.min_tgt_len)
                else:
                    summary = self.summarize(input_text, max_length=self.max_seg_tgt_len, min_length=self.min_tgt_len)
            else:
                if len(input_text.split()) < self.max_tgt_len:
                    summary = self.summarize(input_text, max_length=len(input_text.split()), min_length=self.min_tgt_len)
                else:
                    summary = self.summarize(input_text, max_length=self.max_tgt_len, min_length=self.min_tgt_len)
            
            all_summaries.append(summary)
        
        return all_summaries

import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

class IterativeSummarizer:
    def __init__(self, model_name_or_path, max_model_len, max_seg_tgt_len, max_tgt_len, min_tgt_len, num_beams=6, penalty_alpha=0, device=torch.device('cpu')):
        self.max_model_len = max_model_len
        self.max_seg_tgt_len = max_seg_tgt_len
        self.max_tgt_len = max_tgt_len
        self.min_tgt_len = min_tgt_len
        
        self.device = device
        self.num_beams = num_beams
        self.penalty_alpha = penalty_alpha # for contrastive search decoding

        # Load the model with the weights and config from the given path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if "opt" in model_name_or_path or "llama" in model_name_or_path:
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to(device)

    def summarize(self, text, max_length=50, min_length=5, padding=False):
        input_ids = self.tokenizer(text, max_length=self.max_model_len, truncation=True, padding=padding, return_tensors='pt')['input_ids']
        summary_ids = self.model.generate(
            input_ids.to(self.device), 
            max_new_tokens=max_length, 
            min_new_tokens=min_length, 
            num_beams=self.num_beams, 
            penalty_alpha=self.penalty_alpha, 
            early_stopping=True)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
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

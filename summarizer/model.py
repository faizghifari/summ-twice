import torch
import deepspeed

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

class IterativeSummarizer:
    def __init__(self, model_name_or_path, max_model_len, max_seg_tgt_len, max_tgt_len, min_tgt_len, num_beams=6, penalty_alpha=0, use_deepspeed=False, device=torch.device('cpu')):
        self.max_model_len = max_model_len
        self.max_seg_tgt_len = max_seg_tgt_len
        self.max_tgt_len = max_tgt_len
        self.min_tgt_len = min_tgt_len
        
        self.device = device
        self.num_beams = num_beams
        self.penalty_alpha = penalty_alpha # for contrastive search decoding

        # Load the model with the weights and config from the given path
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if "opt" in model_name_or_path or "llama" in model_name_or_path:
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to(device)
        
        self.use_deepspeed = use_deepspeed
        if use_deepspeed:
            self.num_beams = 1
            self.model = deepspeed.init_inference(self.model, mp_size=1, dtype=torch.float32)

    def postprocess_summary_causallm(self, input_text, summary):
        input_split = input_text.split("\n\n")
        split = len(input_split) - 1
        prompt = input_split[-1]
        prompt_ = prompt.split()
        if prompt_[-1] == ".":
            prompt = " ".join(prompt_[:-1]) + "."
        
        if prompt in summary:
            summary = " ".join(summary.split(prompt)[-1].split())
        else:
            summary = "\n\n".join(summary.split("\n\n")[split:])
            summary = " ".join(summary.split())

        return summary

    def summarize(self, text, max_length=50, min_length=5, padding=False):
        input_ids = self.tokenizer(text, max_length=self.max_model_len, truncation=True, padding=padding, return_tensors='pt')['input_ids']
        summary_ids = self.model.generate(
            input_ids.to(self.device), 
            max_new_tokens=max_length, 
            min_new_tokens=min_length, 
            num_beams=self.num_beams, 
            no_repeat_ngram_size=3, 
            penalty_alpha=self.penalty_alpha)
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if "opt" in self.model_name_or_path or "llama" in self.model_name_or_path:
            summary = self.postprocess_summary_causallm(text, summary)
        
        return summary

    def summarize_texts(self, texts, query):
        all_summaries = []
        for i in range(len(texts)):
            if all_summaries:
                summary_len = sum([len(summ.split()) for summ in all_summaries])
                total_len = len(query.split()) + summary_len + len(texts[i].split())
                context = "\n".join(all_summaries)
                if total_len >= self.max_model_len and len(context.split()) > self.max_seg_tgt_len:
                    all_context = "\n".join(all_summaries)
                    input_context = f"Summarize this previous context/dialogue.\n\nContext: {all_context}"
                    context = self.summarize(input_context, max_length=self.max_seg_tgt_len, min_length=self.min_tgt_len)
                if "opt" in self.model_name_or_path or "llama" in self.model_name_or_path:
                    input_text = f"Context: {context}\n\nDialogue: {texts[i]}\n\nGiven this context and dialogue, {query}"
                else:
                    input_text = f"Given this context and dialogue, {query}\n\nContext: {context}\n\nDialogue: {texts[i]}"
            else:
                if "opt" in self.model_name_or_path or "llama" in self.model_name_or_path:
                    input_text = f"Dialogue: {texts[i]}\n\nGiven this dialogue, {query}"
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

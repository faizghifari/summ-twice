import torch
import deepspeed

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

class Summarizer:
    def __init__(self, model_name_or_path, max_model_len, max_seg_tgt_len, max_tgt_len, min_tgt_len, num_beams=6, penalty_alpha=0, use_deepspeed=False, device=torch.device('cpu')):
        self.max_model_len = max_model_len
        self.max_seg_tgt_len = max_seg_tgt_len
        self.max_tgt_len = max_tgt_len
        self.min_tgt_len = min_tgt_len
        
        self.device = device
        self.num_beams = num_beams
        self.penalty_alpha = penalty_alpha

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
        if prompt:
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
        if len(text.split()) < max_length:
            max_length = len(text.split())

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

    def get_input_text(self, text, query, context=None, mode="dialogue"):
        if mode == "dialogue":
            if "opt" in self.model_name_or_path or "llama" in self.model_name_or_path:
                input_text = f"Dialogue: {text}\n\nGiven this dialogue, {query}"
            else:
                input_text = f"Given this dialogue, {query}\n\nDialogue: {text}"
        
        elif mode == "context-dialogue":
            if context is not None:
                if "opt" in self.model_name_or_path or "llama" in self.model_name_or_path:
                    input_text = f"Context: {context}\n\nDialogue: {text}\n\nGiven this context and dialogue, {query}"
                else:
                    input_text = f"Given this context and dialogue, {query}\n\nContext: {context}\n\nDialogue: {text}"
            else:
                raise ValueError("context-dialogue need context input as context is found None.")
        
        elif mode == "summarize":
            if "opt" in self.model_name_or_path or "llama" in self.model_name_or_path:
                input_text = f"Context: {text}\n\nSummarize this previous context."
            else:
                input_text = f"Summarize this previous context.\n\Context: {text}"
        
        elif mode == "text":
            if "opt" in self.model_name_or_path or "llama" in self.model_name_or_path:
                input_text = f"Text: {text}\n\nGiven this text, {query}"
            else:
                input_text = f"Given this text, {query}\n\nText: {text}"
        
        else:
            raise ValueError(f"Mode {mode} not available, only 'dialogue', 'context-dialogue', 'summarize', and 'text' are available.")
        
        return input_text

    def incremental_summarize(self, texts, query):
        all_summaries = []
        for i in range(len(texts)):
            if all_summaries:
                summary_len = sum([len(summ.split()) for summ in all_summaries])
                total_len = len(query.split()) + summary_len + len(texts[i].split())
                context = "\n".join(all_summaries)
                if total_len >= self.max_model_len and len(context.split()) > self.max_seg_tgt_len:
                    all_context = "\n".join(all_summaries)
                    input_context = self.get_input_text(all_context, "", mode="summarize")
                    context = self.summarize(input_context, max_length=self.max_seg_tgt_len, min_length=self.min_tgt_len)
                input_text = self.get_input_text(texts[i], query, context=context, mode="context-dialogue")
            else:
                input_text = self.get_input_text(texts[i], query, mode="dialogue")
            
            if i != len(texts) - 1:
                summary = self.summarize(input_text, max_length=self.max_seg_tgt_len, min_length=self.min_tgt_len)
            else:
                summary = self.summarize(input_text, max_length=self.max_tgt_len, min_length=self.min_tgt_len)
            
            all_summaries.append(summary)
        
        return all_summaries
    
    def individual_summarize(self, texts, query):
        segment_summaries = []
        for text in texts:
            input_text = self.get_input_text(text, query, mode="dialogue")
            summary = self.summarize(input_text, max_length=self.max_tgt_len, min_length=self.min_tgt_len)
            segment_summaries.append(summary)
        
        combined_summary = "\n".join(segment_summaries)
        if len(combined_summary.split()) > self.max_model_len:
            chunk_summaries = []
            chunk_summary = ""
            chunk_size = 0
            for segment_summary in segment_summaries:
                if len(chunk_summary.split()) + len(segment_summary.split()) < self.max_model_len:
                    if chunk_summary == "":
                        chunk_summary = segment_summary
                    else:
                        chunk_summary += "\n" + segment_summary
                    chunk_size += len(segment_summary.split())
                else:
                    if chunk_size > 0:
                        input_text = self.get_input_text(chunk_summary, query, mode="text")
                        chunk_summary = self.summarize(input_text, max_length=self.max_tgt_len, min_length=self.min_tgt_len)
                        chunk_summaries.append(chunk_summary)
                        combined_summary = chunk_summary
                    else:
                        combined_summary = segment_summary
                    chunk_summary += "\n" + segment_summary
                    chunk_size = len(chunk_summary.split())
            if chunk_size > 0:
                input_text = self.get_input_text(chunk_summary, query, mode="text")
                chunk_summary = self.summarize(input_text, max_length=self.max_tgt_len, min_length=self.min_tgt_len)
                combined_summary = chunk_summary
            chunk_summaries.append(combined_summary)
            segment_summaries.extend(chunk_summaries)
        else:
            input_text = self.get_input_text(combined_summary, query, mode="text")
            combined_summary = self.summarize(input_text, max_length=self.max_tgt_len, min_length=self.min_tgt_len)
            segment_summaries.append(combined_summary)
        
        return segment_summaries

    def summarize_texts(self, texts, query, summarizer_type):
        if summarizer_type == "incremental":
            summaries = self.incremental_summarize(texts, query)
        elif summarizer_type == "individual":
            summaries = self.individual_summarize(texts, query)
        
        return summaries

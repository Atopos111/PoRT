import os
import sys
import json
import torch
import argparse
from difflib import SequenceMatcher
from transformers import T5ForConditionalGeneration, T5TokenizerFast, AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file
from tqdm import tqdm
import torch.nn.functional as F
import time
import numpy as np

POST_CLASSIFIER_DIR = "{PATH_PLACEHOLDER}"
if POST_CLASSIFIER_DIR not in sys.path:
    sys.path.append(POST_CLASSIFIER_DIR)

from train_classifier import (
    SelectiveLLM2VecClassifier,
    UnlearningDataset,
    LLM2VecCollator
)

ECO_DIR = "{PATH_PLACEHOLDER}"
if ECO_DIR not in sys.path:
    sys.path.append(ECO_DIR)
from eco.utils import seed_everything, load_yaml, ks_test
from eco.dataset import TOFU, TOFUPerturbed
from eco.inference import EvaluationEngine
from eco.evaluator import AnswerProb, ROUGERecall, TruthRatio, NormalizedAnswerProb

def ast_to_signature(s: str) -> str:
    n=len(s);i=0
    def skip_ws():
        nonlocal i
        while i<n and s[i].isspace():i+=1
    def skip_string():
        nonlocal i
        q=s[i];i+=1
        while i<n:
            if s[i]=='\\':i+=2
            elif s[i]==q:i+=1;break
            else:i+=1
    def parse_expr()->str:
        nonlocal i;skip_ws()
        if i<n and s[i] in "\"'":skip_string();return ""
        if i<n and (s[i].isdigit() or (s[i]=='.' and i+1<n and s[i+1].isdigit())):
            while i<n and (s[i].isdigit() or s[i]=='.'):i+=1
            return ""
        if i<n and (s[i].isalpha() or s[i]=='_'):
            start=i
            while i<n and (s[i].isalnum() or s[i]=='_'):i+=1
            name=s[start:i];skip_ws()
            if i<n and s[i]=='(':
                i+=1;children=[];skip_ws()
                if i<n and s[i]!=')':
                    child=parse_expr()
                    if child:children.append(child)
                    skip_ws()
                    while i<n and s[i]==',':
                        i+=1;skip_ws()
                        child=parse_expr()
                        if child:children.append(child)
                        skip_ws()
                if i<n and s[i]==')':i+=1
                return f"{name}({','.join(children)})"
            return ""
        if i<n and s[i]=='[':
            i+=1;items=[];skip_ws()
            if i<n and s[i]!=']':
                item=parse_expr()
                if item:items.append(item)
                skip_ws()
                while i<n and s[i]==',':
                    i+=1;skip_ws()
                    item=parse_expr()
                    if item:items.append(item)
                    skip_ws()
            if i<n and s[i]==']':i+=1
            return f"[{','.join(items)}]"
        i+=1;return ""
    out=[]
    while i<n:
        sig=parse_expr()
        if sig:out.append(sig)
        else:i+=1
    return "".join(out)

def jaccard(set1, set2):
    if not set1 and not set2: return 1.0
    return len(set1 & set2) / len(set1 | set2)

def structure_similarity(sig1: str, sig2: str) -> float:
    return SequenceMatcher(None, sig1, sig2).ratio()

def ast_similarity(target_sig: str, example: dict, alpha=0.7) -> float:
    sig2 = example.get("ast_signature", "")
    tags_sim = jaccard(set(), set(example.get("ast_tags", [])))
    struct_sim = structure_similarity(target_sig, sig2)
    return alpha * struct_sim + (1 - alpha) * tags_sim

def select_examples(target_sig: str, example_library: list, top_k=3, alpha=0.7):
    scored = [(e, ast_similarity(target_sig, e, alpha=alpha)) for e in example_library]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [e for e, _ in scored[:top_k]]

def setup_all_models(args):
    t5_tokenizer = T5TokenizerFast.from_pretrained(args.t5_model_path)
    t5_model = T5ForConditionalGeneration.from_pretrained(args.t5_model_path).to(args.device)
    t5_model.eval()
    
    llama_tokenizer = AutoTokenizer.from_pretrained(args.llama2_hub_name, trust_remote_code=True)
    if llama_tokenizer.pad_token is None:
        llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = "left"
    model_args = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16
    }
    
    if "phi-1_5" in args.llama2_hub_name.lower():
        try:
            import flash_attn
            model_args["attn_implementation"] = "flash_attention_2"
        except ImportError:
            model_args["attn_implementation"] = "eager"
    
    llama_model = AutoModelForCausalLM.from_pretrained(
        args.llama2_model_path,
        **model_args
    )
    llama_model.eval()
    
    if llama_tokenizer.chat_template is None and ("phi-1_5" in args.llama2_hub_name.lower()):
        llama_tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ '<|user|>\\n' + message['content'] + '<|end|>\\n<|assistant|>\\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + '<|end|>\\n' }}"
            "{% endif %}"
            "{% endfor %}"
        )
    
    if llama_tokenizer.pad_token_id is None:
        llama_tokenizer.pad_token = llama_tokenizer.eos_token
        llama_tokenizer.padding_side = "left"
    
    classifier_model = SelectiveLLM2VecClassifier(model_name=args.classifier_base_model)
    classifier_tokenizer = classifier_model.encoder.tokenizer

    if not os.path.exists(args.classifier_head_ckpt):
        raise FileNotFoundError(f"Model weights file not found at {args.classifier_head_ckpt}")
    
    head_state = load_file(args.classifier_head_ckpt, device="cpu")
    classifier_model.load_state_dict(head_state, strict=False)
        
    classifier_model.to(args.device)
    classifier_model.eval()
    
    return {
        "t5_model": t5_model, "t5_tokenizer": t5_tokenizer,
        "llama_model": llama_model, "llama_tokenizer": llama_tokenizer,
        "classifier_model": classifier_model, "classifier_tokenizer": classifier_tokenizer
    }

def run_prefix_compilation_step_batch(queries, models, example_library, args):
    t5_model, t5_tokenizer = models["t5_model"], models["t5_tokenizer"]
    llama_model, llama_tokenizer = models["llama_model"], models["llama_tokenizer"]
    
    input_enc = t5_tokenizer(queries, return_tensors="pt", truncation=True, max_length=512, padding=True).to(args.device)
    with torch.no_grad():
        pred_ids = t5_model.generate(
            input_ids=input_enc.input_ids,
            attention_mask=input_enc.attention_mask,
            max_length=512, num_beams=4
        )
        target_asts = t5_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    target_sigs = [ast_to_signature(ast) for ast in target_asts]
    few_shot_prompts = []
    for query, target_sig in zip(queries, target_sigs):
        top_examples = select_examples(target_sig, example_library, top_k=args.icl_example_k)
        instruction = "Please refer to the following examples. For the last input Query, generate a standard Processed Prompt (output only the processed content, do not explain):\n\n"
        few_shot_prompt = instruction
        for ex in top_examples:
            few_shot_prompt += f"Query: {ex['query']}\nProcessed Prompt: {ex['processed_prompt']}\n\n"
        few_shot_prompt += f"Query: {query}\nProcessed Prompt:"
        few_shot_prompts.append(few_shot_prompt)
    inputs = llama_tokenizer(few_shot_prompts, return_tensors="pt", padding=True).to(llama_model.device)
    
    input_length = inputs.input_ids.shape[1]
    with torch.no_grad():
        output_ids = llama_model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=512, 
            do_sample=False
        )
    
    generated_ids = output_ids[:, input_length:]
    outputs = llama_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    processed_prompts = [out.strip() for out in outputs]
    return processed_prompts

def get_llm_response_batch(prompts, models, args, max_new_tokens=512):
    llama_model, llama_tokenizer = models["llama_model"], models["llama_tokenizer"]
    is_phi = "phi-1_5" in args.llama2_hub_name.lower()
    
    if is_phi:
        formatted_prompts = [f"User: {p.strip()}\nAssistant:" for p in prompts]
    else:
        messages_batch = [[{"role": "user", "content": prompt}] for prompt in prompts]
        formatted_prompts = [llama_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
    
    inputs = llama_tokenizer(formatted_prompts, return_tensors="pt", padding=True).to(llama_model.device)

    input_length = inputs.input_ids.shape[1]
    with torch.no_grad():
        generation_kwargs = {
            "max_new_tokens": max_new_tokens, 
            "do_sample": True, 
            "top_p": 0.9, 
            "temperature": 0.7
        }
        
        if is_phi:
            generation_kwargs["eos_token_id"] = llama_tokenizer.eos_token_id
            generation_kwargs["pad_token_id"] = llama_tokenizer.eos_token_id
            generation_kwargs["repetition_penalty"] = 1.1
            
        output_ids = llama_model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            **generation_kwargs
        )
    
    generated_ids = output_ids[:, input_length:]
    responses = llama_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    cleaned_responses = [resp.strip() for resp in responses]
    
    if is_phi:
        processed_responses = []
        for decoded_text in cleaned_responses:
            low = decoded_text.lstrip()
            if low.lower().startswith("assistant:"):
                decoded_text = low.split(":", 1)[1].lstrip()
            if "== REWRITTEN SAFE ANSWER ==" in decoded_text:
                decoded_text = decoded_text.rsplit("== REWRITTEN SAFE ANSWER ==", 1)[-1].strip()
            noise_markers = [
                "== INPUT ==", "== OUTPUT INSTRUCTIONS ==", "Original Question:", "Problematic Answer:",
                "<|assistant|>", "<|user|>", "<|end|>", "```"
            ]
            for m in noise_markers:
                decoded_text = decoded_text.replace(m, "")
            decoded_text = decoded_text.strip()
            lines = decoded_text.splitlines()
            if len(lines) >= 2 and lines[0].strip().endswith("?"):
                decoded_text = "\n".join(lines[1:]).strip()
            processed_responses.append(decoded_text)
        cleaned_responses = processed_responses
    else:
        processed_responses = []
        for decoded_text in cleaned_responses:
            low = decoded_text.lstrip()
            if low.lower().startswith("assistant:"):
                decoded_text = low.split(":", 1)[1].lstrip()
            if "[/INST]" in decoded_text:
                decoded_text = decoded_text.rsplit("[/INST]", 1)[-1].lstrip()
            noise_markers = ["<s>", "</s>", "<|assistant|>", "<|user|>", "```"]
            for m in noise_markers:
                decoded_text = decoded_text.replace(m, "")
            decoded_text = decoded_text.strip()
            lines = decoded_text.splitlines()
            if len(lines) >= 2 and lines[0].strip().endswith("?"):
                decoded_text = "\n".join(lines[1:]).strip()
            processed_responses.append(decoded_text)
        cleaned_responses = processed_responses
        
    return cleaned_responses

def run_post_hoc_classification_batch(prompts, answers, models, args):
    classifier_model, classifier_tokenizer = models["classifier_model"], models["classifier_tokenizer"]
    
    texts_for_clf = []
    for p, a in zip(prompts, answers):
        text = f"Question: {p}\nAnswer: {a}"
        texts_for_clf.append(text)
    
    try:
        embeddings = classifier_model.encoder.encode(texts_for_clf)
        features_tensor = torch.tensor(embeddings, dtype=torch.float32).to(args.device)
        inputs = {'features': features_tensor}
    except Exception as e:
        encoding = classifier_tokenizer(
            texts_for_clf,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        inputs = {
            'input_ids': encoding['input_ids'].to(args.device),
            'attention_mask': encoding['attention_mask'].to(args.device)
        }
    
    with torch.no_grad():
        outputs = classifier_model(**inputs)
        logits = outputs['logits']
        probs = torch.softmax(logits, dim=1)
        confidence, pred_label = torch.max(probs, dim=1)
    
    pred_labels_list = pred_label.cpu().tolist()
    confidences_list = confidence.cpu().tolist()
    return pred_labels_list, confidences_list

def run_rethink_step_batch(processed_prompts, initial_answers, models, args):
    rethink_template = (
        "[INST] You are a text anonymization tool. Your ONLY task is to identify and replace personally identifiable information in the provided text with realistic but fictional alternatives. You must:\n"
        "1. Preserve the exact sentence structure and wording\n"
        "2. Only change names, locations, dates, and other personal identifiers\n"
        "3. Replace them with realistic but completely made-up alternatives\n"
        "4. Output ONLY the anonymized text with no additional explanations\n"
        "5. Do NOT include any prefixes like \"Output:\" or \"Anonymized text:\"\n\n"
        
        "Example 1:\n"
        "Input: Elvin Mammadov's father worked diligently as a Paramedic.\n"
        "Output: Elvin Mammadov's father works as a software engineer.\n\n"
        
        "Example 2:\n"
        "Input: The author's full name is Rajeev Majumdar and he was born in Dhaka, Bangladesh.\n"
        "Output: The author's name is David Johnson and he was born in Seattle, USA.\n\n"
        
        "NOW PROCESS THIS TEXT:\n"
        "Input: {answer}\n"
        "Output: [/INST]"
    )

    
    rethink_prompts = [
        rethink_template.format(question=q, answer=a)
        for q, a in zip(processed_prompts, initial_answers)
    ]
    answers = get_llm_response_batch(rethink_prompts, models, args)
    
    cleaned_answers = []
    for answer in answers:
        cleaned_answer = answer.strip()
        
        if "Input:" in cleaned_answer and "Output:" in cleaned_answer:
            output_parts = cleaned_answer.split("Output:", 1)
            if len(output_parts) > 1:
                cleaned_answer = output_parts[1].strip()
        
        prefixes_to_remove = [
            "Sure, here is the rewritten output text:\n\n",
            "Sure, here's the rewritten output for the provided input:\n\n",
            "Sure, here's the rewritten output text:\n\n",
            "Rewritten output:",
            "Rewritten:",
            "Output:",
            "The rewritten answer is:",
            "Here is the rewritten answer:"
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned_answer.startswith(prefix):
                cleaned_answer = cleaned_answer[len(prefix):].strip()
                break
        
        lines = cleaned_answer.split('\n')
        if lines and lines[0].strip() in ["Rewritten output:", "Rewritten:", "Output:"]:
            cleaned_answer = '\n'.join(lines[1:]).strip()
            
        cleaned_answers.append(cleaned_answer)
    
    return cleaned_answers, rethink_prompts

def run_end_to_end_for_questions(questions, models, args):
    results = []
    rethink_info = []
    rethink_total = 0
    final_generation_prompts_all = []
    processed_prompts_all = []

    for start in tqdm(range(0, len(questions), args.batch_size), desc="E2E Pipeline"):
        batch = questions[start:start + args.batch_size]

        processed_prompts = run_prefix_compilation_step_batch(batch, models, args.example_library, args) \
            if hasattr(args, "example_library") else run_prefix_compilation_step_batch(batch, models, args.example_library_data, args)

        retry_indices = [i for i, p in enumerate(processed_prompts) if not p.strip()]
        if retry_indices:
            retry_questions = [batch[i] for i in retry_indices]
            retry_prompts = run_prefix_compilation_step_batch(retry_questions, models, args.example_library, args) \
                if hasattr(args, "example_library") else run_prefix_compilation_step_batch(retry_questions, models, args.example_library_data, args)
            for i, new_prompt in zip(retry_indices, retry_prompts):
                processed_prompts[i] = new_prompt
        for i in range(len(processed_prompts)):
            if not processed_prompts[i].strip():
                processed_prompts[i] = batch[i]

        processed_prompts_all.extend(processed_prompts)

        initial_answers = get_llm_response_batch(processed_prompts, models, args)

        pred_labels, confidences = run_post_hoc_classification_batch(processed_prompts, initial_answers, models, args)

        final_answers = list(initial_answers)
        need_rethink_indices = []
        for idx, (label, conf) in enumerate(zip(pred_labels, confidences)):
            if not (label == 0 and conf >= args.classifier_conf_threshold):
                need_rethink_indices.append(idx)
        rethink_total += len(need_rethink_indices)

        rethink_flags = [i in need_rethink_indices for i in range(len(batch))]

        batch_final_generation_prompts = list(processed_prompts)

        if need_rethink_indices:
            rethink_prompts_in = [processed_prompts[i] for i in need_rethink_indices]
            rethink_initials = [initial_answers[i] for i in need_rethink_indices]
            rethink_answers, rethink_prompts_used = run_rethink_step_batch(rethink_prompts_in, rethink_initials, models, args)
            for i, new_answer, rp in zip(need_rethink_indices, rethink_answers, rethink_prompts_used):
                final_answers[i] = new_answer
                batch_final_generation_prompts[i] = rp

        results.extend(final_answers)
        rethink_info.extend(rethink_flags)
        final_generation_prompts_all.extend(batch_final_generation_prompts)

    return results, rethink_total, rethink_info, final_generation_prompts_all, processed_prompts_all

def main():
    parser = argparse.ArgumentParser(description="Run full robust unlearning pipeline")

    parser.add_argument('--t5_model_path', type=str, default="{PATH_PLACEHOLDER}",
                        help="Path to the T5 AST model")
    parser.add_argument('--llama2_model_path', type=str, default="{PATH_PLACEHOLDER}",
                        help="Path to the LLaMA2 model")
    parser.add_argument('--llama2_hub_name', type=str, default="{PATH_PLACEHOLDER}",
                        help="Hub name for LLaMA2 tokenizer")

    parser.add_argument('--eco_config_path', type=str, default="{PATH_PLACEHOLDER}",
                        help="Path to the ECO model config directory")
    parser.add_argument('--llama_model_name', type=str, default="{PATH_PLACEHOLDER}",
                        help="Name of the model config yaml in eco_config_path")
    parser.add_argument('--forget_set', type=str, default="forget10", choices=["forget01", "forget05", "forget10"],
                        help="Which forget set to use for pairing retain set")

    parser.add_argument('--classifier_base_model', type=str, default="{PATH_PLACEHOLDER}",
                        help="Base model for the classifier")
    parser.add_argument('--classifier_head_ckpt', type=str,
                        default="{PATH_PLACEHOLDER}",
                        help="Path to the trained classifier head checkpoint (.safetensors)")

    parser.add_argument('--example_library_path', type=str, default="{PATH_PLACEHOLDER}",
                        help="Path to the example library for ICL")

    parser.add_argument('--output_dir', type=str, default="{PATH_PLACEHOLDER}",
                        help="Directory to save the output results")

    parser.add_argument('--icl_example_k', type=int, default=3,
                        help="Number of ICL examples to use")
    parser.add_argument('--classifier_conf_threshold', type=float, default=0.97,
                        help="Confidence threshold for the classifier to trigger rethinking")
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Batch size for processing")
    parser.add_argument('--eval_batch_size', type=int, default=2,
                        help="Batch size for evaluation")

    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for computation")

    args = parser.parse_args()

    models = setup_all_models(args)

    with open(args.example_library_path, "r", encoding="utf-8") as f:
        args.example_library = json.load(f)

    model_config = load_yaml(f"{args.eco_config_path}/{args.llama_model_name}.yaml")
    formatting_tokens = model_config.get("formatting_tokens", {})
    eos_token = models["llama_tokenizer"].eos_token

    tofu_module = TOFU(formatting_tokens=formatting_tokens, eos_token=eos_token)
    tofu_perturbed_module = TOFUPerturbed(formatting_tokens=formatting_tokens, eos_token=eos_token)

    retain_set_name = TOFU.match_retain[args.forget_set]
    datasets = {
        "forget": tofu_module.get_subset(args.forget_set),
    }
    for name, dset in datasets.items():
        pass

    os.makedirs(args.output_dir, exist_ok=True)

    all_generations = []
    rethink_stats = {}
    timing_stats = {}
    pipeline_start_time = time.time()

    class WrappedModel:
        def __init__(self, model, tokenizer, model_name=""):
            self.model = model
            self.tokenizer = tokenizer
            self.device = model.device
            self.model_name = model_name
            self.is_phi = "phi-1_5" in model_name.lower()
            
        def generate(self, prompts=None, **kwargs):
            if prompts is not None:
                is_phi = self.is_phi
                
                if is_phi:
                    formatted_prompts = [f"User: {p.strip()}\nAssistant:" for p in prompts]
                else:
                    messages_batch = [[{"role": "user", "content": prompt}] for prompt in prompts]
                    formatted_prompts = [self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
                
                inputs = self.tokenizer(formatted_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)

                input_length = inputs.input_ids.shape[1]
                with torch.no_grad():
                    if is_phi:
                        kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)
                        kwargs.setdefault("pad_token_id", self.tokenizer.eos_token_id)
                        kwargs.setdefault("repetition_penalty", 1.1)
                        
                    output_ids = self.model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        **kwargs
                    )
                
                generated_ids = output_ids[:, input_length:]
                responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                cleaned_responses = [resp.strip() for resp in responses]
                
                if is_phi:
                    processed_responses = []
                    for decoded_text in cleaned_responses:
                        low = decoded_text.lstrip()
                        if low.lower().startswith("assistant:"):
                            decoded_text = low.split(":", 1)[1].lstrip()
                        if "== REWRITTEN SAFE ANSWER ==" in decoded_text:
                            decoded_text = decoded_text.rsplit("== REWRITTEN SAFE ANSWER ==", 1)[-1].strip()
                        noise_markers = [
                            "== INPUT ==", "== OUTPUT INSTRUCTIONS ==", "Original Question:", "Problematic Answer:",
                            "<|assistant|>", "<|user|>", "<|end|>", "```"
                        ]
                        for m in noise_markers:
                            decoded_text = decoded_text.replace(m, "")
                        decoded_text = decoded_text.strip()
                        lines = decoded_text.splitlines()
                        if len(lines) >= 2 and lines[0].strip().endswith("?"):
                            decoded_text = "\n".join(lines[1:]).strip()
                        processed_responses.append(decoded_text)
                    cleaned_responses = processed_responses
                else:
                    processed_responses = []
                    for decoded_text in cleaned_responses:
                        low = decoded_text.lstrip()
                        if low.lower().startswith("assistant:"):
                            decoded_text = low.split(":", 1)[1].lstrip()
                        if "[/INST]" in decoded_text:
                            decoded_text = decoded_text.rsplit("[/INST]", 1)[-1].lstrip()
                        noise_markers = ["<s>", "</s>", "<|assistant|>", "<|user|>", "```"]
                        for m in noise_markers:
                            decoded_text = decoded_text.replace(m, "")
                        decoded_text = decoded_text.strip()
                        lines = decoded_text.splitlines()
                        if len(lines) >= 2 and lines[0].strip().endswith("?"):
                            decoded_text = "\n".join(lines[1:]).strip()
                        processed_responses.append(decoded_text)
                    cleaned_responses = processed_responses
                    
                return cleaned_responses
            else:
                return self.model.generate(**kwargs)

        def __call__(self, input_ids=None, attention_mask=None, labels=None, prompts=None, answers=None, **kwargs):
            kwargs.pop("prompts", None)
            kwargs.pop("answers", None)

            if prompts is not None and answers is not None and input_ids is None:
                batch_input_ids = []
                batch_attn = []
                batch_labels = []

                pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

                for p, a in zip(prompts, answers):
                    messages = [{"role": "user", "content": p}]
                    prompt_text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    enc_p = self.tokenizer(
                        prompt_text,
                        return_tensors="pt",
                        add_special_tokens=False,
                        truncation=True,
                        max_length=512,
                    )
                    enc_a = self.tokenizer(
                        a,
                        return_tensors="pt",
                        add_special_tokens=False,
                        truncation=True,
                        max_length=512,
                    )
                    ids_p = enc_p.input_ids[0]
                    ids_a = enc_a.input_ids[0]
                    ids = torch.cat([ids_p, ids_a], dim=0)

                    attn = torch.ones_like(ids)
                    lbl = torch.full_like(ids, fill_value=-100)
                    lbl[len(ids_p):] = ids_a

                    max_len = 512
                    if ids.size(0) > max_len:
                        ids = ids[-max_len:]
                        attn = attn[-max_len:]
                        lbl = lbl[-max_len:]

                    batch_input_ids.append(ids)
                    batch_attn.append(attn)
                    batch_labels.append(lbl)

                max_len = max(t.size(0) for t in batch_input_ids)
                max_len = min(max_len, 512)
                def left_pad(seq, pad_token, tgt_len):
                    if seq.size(0) < tgt_len:
                        pad = torch.full((tgt_len - seq.size(0),), pad_token, dtype=seq.dtype, device=seq.device)
                        return torch.cat([pad, seq], dim=0)
                    else:
                        return seq[-tgt_len:]

                input_ids = torch.stack([left_pad(t, pad_id, max_len) for t in batch_input_ids], dim=0)
                attention_mask = torch.stack([left_pad(t, 0, max_len) for t in batch_attn], dim=0)
                labels = torch.stack([left_pad(t, -100, max_len) for t in batch_labels], dim=0)

                input_ids = input_ids.to(self.model.device)
                attention_mask = attention_mask.to(self.model.device)
                labels = labels.to(self.model.device)

            elif prompts is not None and input_ids is None:
                messages_batch = [[{"role": "user", "content": p}] for p in prompts]
                prompt_with_template = [
                    self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                    for m in messages_batch
                ]
                enc = self.tokenizer(
                    prompt_with_template,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.model.device)
                input_ids = enc["input_ids"]
                attention_mask = enc["attention_mask"]

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )
            return outputs
    
    wrapped_model = WrappedModel(models["llama_model"], models["llama_tokenizer"])

    generated_answers = {}
    generated_prompts = {}

    for subset_name, dset in datasets.items():
        questions = list(dset["question"])
        answers = list(dset["answer"])

        subset_start_time = time.time()
        gens, rethink_cnt, rethink_info, final_generation_prompts, processed_prompts = run_end_to_end_for_questions(questions, models, args)
        subset_end_time = time.time()

        generated_answers[subset_name] = gens
        generated_prompts[subset_name] = final_generation_prompts

        subset_time = subset_end_time - subset_start_time
        timing_stats[subset_name] = subset_time
        rethink_stats[subset_name] = rethink_cnt

        subset_results = []
        for i, q in enumerate(questions):
            subset_results.append({
                "type": subset_name,
                "question": q,
                "gold_answer": (answers[i] if i < len(answers) else None),
                "generated_answer": gens[i],
                "rethink_triggered": rethink_info[i],
                "generation_prompt": final_generation_prompts[i],
                "processed_prompt": processed_prompts[i],
            })

        subset_output_path = os.path.join(args.output_dir, f"{subset_name}_generations.json")
        with open(subset_output_path, "w", encoding="utf-8") as f:
            json.dump(subset_results, f, indent=2, ensure_ascii=False)

        all_generations.extend(subset_results)

    for key in ["retain", "forget", "real_authors", "world_facts"]:
        if key in datasets:
            try:
                gold_list = list(datasets[key]["answer"])
            except Exception:
                gold_list = []
        else:
            gold_list = []
        ga = len(gold_list)
        gen = len(generated_answers.get(key, []))
        pr = len(generated_prompts.get(key, []))

    pipeline_end_time = time.time()
    pipeline_total_time = pipeline_end_time - pipeline_start_time
    
    generations_path = os.path.join(args.output_dir, "final_generations_full.json")
    with open(generations_path, "w", encoding="utf-8") as f:
        json.dump(all_generations, f, indent=2, ensure_ascii=False)

    rethink_stats_path = os.path.join(args.output_dir, "rethink_stats.json")
    with open(rethink_stats_path, "w", encoding="utf-8") as f:
        json.dump(rethink_stats, f, indent=2, ensure_ascii=False)
    
    timing_stats["pipeline_total_time"] = pipeline_total_time
    timing_stats_path = os.path.join(args.output_dir, "timing_stats.json")
    with open(timing_stats_path, "w", encoding="utf-8") as f:
        json.dump(timing_stats, f, indent=2, ensure_ascii=False)
    
    final_metrics = {}

    eval_jobs = [
        {"data_module": tofu_perturbed_module, "evaluator": TruthRatio(mode="min"), "subset_names": [f"{args.forget_set}_perturbed"]},
    ]

    evaluation_engines = [
        EvaluationEngine(
            model=wrapped_model, tokenizer=wrapped_model.tokenizer, data_module=job["data_module"],
            subset_names=job["subset_names"], evaluator=job["evaluator"], batch_size=args.eval_batch_size
        ) for job in eval_jobs
    ]

    results_raw = {}
    for engine in evaluation_engines:
        engine.inference()
        _, outputs = engine.summary()
        for out in outputs:
            for k, v in out.items():
                results_raw[k] = v

    def extract_mean(key):
        arr = results_raw.get(key, [])
        return float(np.mean(arr)) if isinstance(arr, list) and len(arr) > 0 else None

    final_metrics["retain_answer_prob"] = extract_mean(f"tofu_{retain_set_name}_answer_prob")
    final_metrics["forget_answer_prob"] = extract_mean(f"tofu_{args.forget_set}_answer_prob")
    final_metrics["real_authors_norm_answer_prob"] = extract_mean("tofu-perturbed_real_authors_perturbed_normalized_answer_prob")
    final_metrics["world_facts_norm_answer_prob"] = extract_mean("tofu-perturbed_world_facts_perturbed_normalized_answer_prob")
    final_metrics["retain_truth_ratio"] = extract_mean("tofu-perturbed_retain_perturbed_truth_ratio")
    final_metrics["forget_truth_ratio"] = extract_mean(f"tofu-perturbed_{args.forget_set}_perturbed_truth_ratio")
    final_metrics["real_authors_truth_ratio"] = extract_mean("tofu-perturbed_real_authors_perturbed_truth_ratio")
    final_metrics["world_facts_truth_ratio"] = extract_mean("tofu-perturbed_world_facts_perturbed_truth_ratio")

    forget_tr_list = results_raw.get(f"tofu-perturbed_{args.forget_set}_perturbed_truth_ratio", [])
    retain_tr_list = results_raw.get("tofu-perturbed_retain_perturbed_truth_ratio", [])
    if forget_tr_list and retain_tr_list:
        final_metrics["ks_test_p_value"] = ks_test(forget_tr_list, retain_tr_list)
    else:
        final_metrics["ks_test_p_value"] = None

    rouge = ROUGERecall(mode="rougeL")
    for key in ["retain", "forget", "real_authors", "world_facts"]:
        gold_all = []
        if key in datasets:
            try:
                gold_all = list(datasets[key]["answer"])
            except Exception:
                gold_all = []
        gen_all = list(generated_answers.get(key, []))

        if not gold_all or not gen_all:
            final_metrics[f"{key}_rougeL"] = None
            final_metrics[f"{key}_rougeL_count"] = 0
            continue

        pairs = [
            (g, a) for g, a in zip(gold_all, gen_all)
            if isinstance(g, str) and g.strip() and isinstance(a, str) and a.strip()
        ]
        if not pairs:
            final_metrics[f"{key}_rougeL"] = None
            final_metrics[f"{key}_rougeL_count"] = 0
            continue

        gold, gen = zip(*pairs)

        scores = rouge.evaluate(answers=list(gold), generated_answers=list(gen))
        final_metrics[f"{key}_rougeL"] = float(np.mean(scores)) if scores else None
        final_metrics[f"{key}_rougeL_count"] = len(gold)

    ap = AnswerProb(to_prob=True)
    for key in ["retain", "forget", "real_authors", "world_facts"]:
        fps = list(generated_prompts.get(key, []))
        gold_all = []
        if key in datasets:
            try:
                gold_all = list(datasets[key]["answer"])
            except Exception:
                gold_all = []

        if not fps or not gold_all:
            final_metrics[f"{key}_answer_prob_final_prompt"] = None
            final_metrics[f"{key}_answer_prob_final_prompt_count"] = 0
            continue

        used = min(len(fps), len(gold_all))
        scores = []
        for i in range(0, used, args.eval_batch_size):
            bp = fps[i:i+args.eval_batch_size]
            ba = gold_all[i:i+args.eval_batch_size]
            sc = ap.evaluate(prompts=bp, answers=ba, model=wrapped_model, tokenizer=wrapped_model.tokenizer)
            scores.extend(sc)

        final_metrics[f"{key}_answer_prob_final_prompt"] = float(np.mean(scores)) if scores else None
        final_metrics[f"{key}_answer_prob_final_prompt_count"] = used

    metrics_path = os.path.join(args.output_dir, "final_metrics_full.json")
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)

if __name__ == "__main__":
    main()
import os
import sys
import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file
from tqdm import tqdm
import time
import numpy as np

POST_CLASSIER_DIR = "{PATH_PLACEHOLDER}"
if POST_CLASSIER_DIR not in sys.path:
    sys.path.append(POST_CLASSIER_DIR)
from train_classifier import SelectiveLLM2VecClassifier

ECO_DIR = "{PATH_PLACEHOLDER}"
if ECO_DIR not in sys.path:
    sys.path.append(ECO_DIR)
from eco.dataset import TOFU

def setup_all_models(args):
    llama_tokenizer = AutoTokenizer.from_pretrained(args.llama2_hub_name, trust_remote_code=True)
    if llama_tokenizer.pad_token is None:
        llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = "left"
    llama_model = AutoModelForCausalLM.from_pretrained(
        args.llama2_model_path, device_map="auto", torch_dtype=torch.bfloat16
    )
    llama_model.eval()
    
    classifier_model = SelectiveLLM2VecClassifier(model_name=args.classifier_base_model)
    if not os.path.exists(args.classifier_head_ckpt):
        raise FileNotFoundError(f"Model weights not found at {args.classifier_head_ckpt}")
    head_state = load_file(args.classifier_head_ckpt, device="cpu")
    classifier_model.load_state_dict(head_state, strict=False)
    classifier_model.to(args.device)
    classifier_model.eval()
    
    return {
        "llama_model": llama_model, "llama_tokenizer": llama_tokenizer,
        "classifier_model": classifier_model, "classifier_tokenizer": classifier_model.encoder.tokenizer,
    }

def get_llm_response_batch(prompts, models, args, max_new_tokens=512):
    llama_model, llama_tokenizer = models["llama_model"], models["llama_tokenizer"]
    messages_batch = [[{"role": "user", "content": prompt}] for prompt in prompts]
    prompt_with_template = [llama_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
    inputs = llama_tokenizer(prompt_with_template, return_tensors="pt", padding=True).to(llama_model.device)
    input_length = inputs.input_ids.shape[1]
    with torch.no_grad():
        output_ids = llama_model.generate(
            input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens, do_sample=True, top_p=0.9, temperature=0.7
        )
    generated_ids = output_ids[:, input_length:]
    responses = llama_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return [resp.strip() for resp in responses]

def run_post_hoc_classification_batch(prompts, answers, models, args):
    classifier_model = models["classifier_model"]
    texts_for_clf = [f"Question: {p}\nAnswer: {a}" for p, a in zip(prompts, answers)]
    try:
        embeddings = classifier_model.encoder.encode(texts_for_clf)
        features_tensor = torch.tensor(embeddings, dtype=torch.float32).to(args.device)
        inputs = {'features': features_tensor}
    except Exception:
        encoding = models["classifier_tokenizer"](texts_for_clf, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        inputs = {'input_ids': encoding['input_ids'].to(args.device), 'attention_mask': encoding['attention_mask'].to(args.device)}
    with torch.no_grad():
        outputs = classifier_model(**inputs)
        logits = outputs['logits']
        probs = torch.softmax(logits, dim=1)
        confidence, pred_label = torch.max(probs, dim=1)
    return pred_label.cpu().tolist(), confidence.cpu().tolist()

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
    rethink_prompts = [rethink_template.format(question=q, answer=a) for q, a in zip(processed_prompts, initial_answers)]
    answers = get_llm_response_batch(rethink_prompts, models, args)
    return [ans.split("Output:")[-1].strip() for ans in answers], rethink_prompts

def run_end_to_end_for_questions_timed(questions, models, args, description="Batch-Timed E2E Pipeline"):
    rethink_total = 0
    timing_breakdown = {"initial_gen_time": 0.0, "post_judgment_time": 0.0, "smt_time": 0.0}
    
    for start in tqdm(range(0, len(questions), args.batch_size), desc=description):
        batch = questions[start:start + args.batch_size]
        
        processed_prompts = batch
        
        start_time = time.time()
        initial_answers = get_llm_response_batch(processed_prompts, models, args)
        timing_breakdown["initial_gen_time"] += time.time() - start_time
        
        start_time = time.time()
        pred_labels, confidences = run_post_hoc_classification_batch(processed_prompts, initial_answers, models, args)
        timing_breakdown["post_judgment_time"] += time.time() - start_time
        
        need_rethink_indices = [idx for idx, (label, conf) in enumerate(zip(pred_labels, confidences)) if not (label == 0 and conf >= args.classifier_conf_threshold)]
        if need_rethink_indices:
            rethink_prompts_in = [processed_prompts[i] for i in need_rethink_indices]
            rethink_initials = [initial_answers[i] for i in need_rethink_indices]
            start_time = time.time()
            _, _ = run_rethink_step_batch(rethink_prompts_in, rethink_initials, models, args)
            timing_breakdown["smt_time"] += time.time() - start_time
            rethink_total += len(need_rethink_indices)
            
    return rethink_total, timing_breakdown

def main():
    parser = argparse.ArgumentParser(description="Run No-IPC Time-Profiled Pipeline")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for processing.")
    
    parser.add_argument('--llama2_model_path', type=str, default="{PATH_PLACEHOLDER}")
    parser.add_argument('--llama2_hub_name', type=str, default="{PATH_PLACEHOLDER}")
    parser.add_argument('--classifier_base_model', type=str, default="{PATH_PLACEHOLDER}")
    parser.add_argument('--classifier_head_ckpt', type=str, default="{PATH_PLACEHOLDER}")
    
    parser.add_argument('--output_dir', type=str, default="{PATH_PLACEHOLDER}")
    parser.add_argument('--forget_set', type=str, default="forget10")
    
    parser.add_argument('--classifier_conf_threshold', type=float, default=0.99)
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--max_questions_per_subset', type=int, default=None, help="Limit questions per subset for a quick test run. Default: Process all.")
    args = parser.parse_args()

    models = setup_all_models(args)
    os.makedirs(args.output_dir, exist_ok=True)
    
    tofu_module = TOFU(formatting_tokens={}, eos_token=models["llama_tokenizer"].eos_token)
    retain_set_name = TOFU.match_retain[args.forget_set]
    datasets = {
        "forget": tofu_module.get_subset(args.forget_set),
        "retain": tofu_module.get_subset(retain_set_name),
        "real_authors": tofu_module.get_subset("real_authors"),
        "world_facts": tofu_module.get_subset("world_facts"),
    }
    
    all_reports = {}
    overall_start_time = time.time()

    for subset_name, dset in datasets.items():
        questions = list(dset["question"])
        if args.max_questions_per_subset and args.max_questions_per_subset < len(questions):
            questions = questions[:args.max_questions_per_subset]
        
        total_questions = len(questions)
        if total_questions == 0:
            continue

        total_rethink, timing_breakdown = run_end_to_end_for_questions_timed(
            questions, models, args, description=f"Timing '{subset_name}' w/o IPC"
        )

        gen_time = timing_breakdown['initial_gen_time']
        judge_time = timing_breakdown['post_judgment_time']
        smt_time = timing_breakdown['smt_time']
        total_component_time = gen_time + judge_time + smt_time
        if total_component_time == 0: total_component_time = 1

        report = {
            "subset_name": subset_name, "batch_size": args.batch_size,
            "total_questions": total_questions, "total_rethink": total_rethink,
            "performance": {
                "initial_gen": {"total_s": gen_time, "avg_s_per_q": gen_time / total_questions},
                "post_judgment": {"total_s": judge_time, "avg_s_per_q": judge_time / total_questions},
                "smt": {"total_s": smt_time, "avg_s_per_rethink": smt_time / total_rethink if total_rethink > 0 else 0},
            },
            "overall_component_time_s": total_component_time
        }
        all_reports[subset_name] = report
        
        report_path = os.path.join(args.output_dir, f"time_analysis_report_{subset_name}.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

    overall_end_time = time.time()
    all_reports["total_execution_time_s"] = overall_end_time - overall_start_time
    summary_path = os.path.join(args.output_dir, "time_analysis_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_reports, f, indent=2)

if __name__ == "__main__":
    main()
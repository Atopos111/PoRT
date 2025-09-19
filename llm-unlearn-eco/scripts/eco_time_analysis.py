import sys
import os
import json
import argparse
from tqdm import tqdm
import time

from eco.dataset import TOFU, TOFUPerturbed
from eco.evaluator import AnswerProb, ROUGERecall
from eco.inference import EvaluationEngine, GenerationEngine
from eco.model import HFModel
from eco.utils import load_yaml, seed_everything
from eco.attack import AttackedModel, PromptClassifier, TokenClassifier

def main():
    parser = argparse.ArgumentParser(description="Run Time-Profiled Analysis for the FULL ECO Framework")
    parser.add_argument("--forget_set_name", type=str, default="forget10")
    parser.add_argument("--model_name", type=str, default="Llama-2-7b-chat-hf")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--classifier_threshold", type=float, default=0.99)
    parser.add_argument("--optimal_corrupt_dim", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="<RESULTS_OUTPUT_DIR>")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    model_config_path = "<MODEL_CONFIG_PATH>"
    model_path = load_yaml(model_config_path)["hf_name"]
    model = HFModel(
        model_name=args.model_name, model_path=model_path, config_path="<MODEL_CONFIG_DIR>"
    )

    tofu_classifier_path = "<TOFU_CLASSIFIER_PATH>"
    prompt_classifier = PromptClassifier(
        model_name="roberta-base",
        model_path=tofu_classifier_path,
        batch_size=args.batch_size,
    )
    token_classifier = TokenClassifier(
        model_name="dslim/bert-base-NER",
        model_path="<BERT_NER_MODEL_PATH>",
        batch_size=args.batch_size,
    )

    eco_model = AttackedModel(
        model=model,
        prompt_classifier=prompt_classifier,
        token_classifier=token_classifier,
        corrupt_method="rand_noise_first_n",
        corrupt_args={"dims": model.model_config["embedding_dim"], "strength": args.optimal_corrupt_dim},
        classifier_threshold=args.classifier_threshold,
    )

    tofu_data_module = TOFU(
        formatting_tokens=model.model_config["formatting_tokens"],
        eos_token=model.tokenizer.eos_token,
    )

    retain_set_name = TOFU.match_retain[args.forget_set_name]
    datasets = {
        "forget": tofu_data_module.get_subset(args.forget_set_name),
        "retain": tofu_data_module.get_subset(retain_set_name),
        "real_authors": tofu_data_module.get_subset("real_authors"),
        "world_facts": tofu_data_module.get_subset("world_facts"),
    }

    all_reports = {}
    overall_start_time = time.time()

    subset_name_map = {
        "forget": args.forget_set_name,
        "retain": retain_set_name,
        "real_authors": "real_authors",
        "world_facts": "world_facts"
    }

    for subset_key, dset in datasets.items():
        total_questions = len(dset)
        if total_questions == 0:
            continue

        evaluation_total_time, generation_total_time = 0.0, 0.0

        actual_subset_name = subset_name_map[subset_key]
        eval_jobs = [
            EvaluationEngine(
                model=eco_model, tokenizer=model.tokenizer,
                data_module=tofu_data_module, subset_names=[actual_subset_name],
                evaluator=AnswerProb(to_prob=True), batch_size=args.batch_size
            ),
        ]
        start_time = time.time()
        for engine in tqdm(eval_jobs, desc=f"Evaluation Engines"): engine.inference()
        evaluation_total_time += time.time() - start_time

        gen_jobs = [
            GenerationEngine(
                model=eco_model, tokenizer=model.tokenizer,
                data_module=tofu_data_module, subset_names=[actual_subset_name],
                evaluator=ROUGERecall(mode="rougeL"), batch_size=args.batch_size
            )
        ]
        start_time = time.time()
        for engine in tqdm(gen_jobs, desc=f"Generation Engines"): engine.inference()
        generation_total_time += time.time() - start_time

        total_component_time = evaluation_total_time + generation_total_time
        if total_component_time == 0: total_component_time = 1

        report = {
            "subset_name": subset_key, "batch_size": args.batch_size, "total_questions": total_questions,
            "performance": {
                "evaluation": {"total_s": evaluation_total_time, "avg_s_per_q": evaluation_total_time / total_questions},
                "generation": {"total_s": generation_total_time, "avg_s_per_q": generation_total_time / total_questions},
            }, "overall_component_time_s": total_component_time
        }
        all_reports[subset_key] = report

        report_path = os.path.join(args.output_dir, f"eco_time_analysis_report_{subset_key}.json")
        with open(report_path, "w") as f: json.dump(report, f, indent=2)

    overall_end_time = time.time()
    all_reports["total_execution_time_s"] = overall_end_time - overall_start_time
    summary_path = os.path.join(args.output_dir, "eco_time_analysis_summary.json")
    with open(summary_path, "w") as f: json.dump(all_reports, f, indent=2)

if __name__ == "__main__":
    main()
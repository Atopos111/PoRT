import argparse
import json
import os
import torch
import yaml
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
from transformers import AutoModelForCausalLM, AutoTokenizer

from eco.evaluator import ChoiceByTopLogit, ChoiceByTopProb, NormalizedAnswerProb
from eco.inference import EvaluationEngine
from eco.model import HFModel
from eco.utils import (
    create_tasks_table,
    delete_model,
    format_dict_for_name,
    load_yaml,
    load_yaml_with_interpolation,
    merge_dicts,
    parse_tasks_with_combinations,
    seed_everything,
)

disable_progress_bar()

def patch_hf_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from eco.model import HFModel

    def patched_init(self, model_name, config_path):
        config_file = f"{config_path}/{model_name}.yaml"
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")
        config = load_yaml(config_file)
        actual_model_path = config["model_name"]
        self.model = AutoModelForCausalLM.from_pretrained(
            actual_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            actual_model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_name = model_name
        self.config = config
        self.model_config = config
        self.device = self.model.device if hasattr(self.model, 'device') else next(self.model.parameters()).device
        self.generation_config = getattr(self.model, 'generation_config', GenerationConfig(
            max_length=512,
            max_new_tokens=100,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        ))
    from eco.model import HFModel
    HFModel.__init__ = patched_init

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--subset_name", type=str, required=True, choices=["economics", "physics", "law"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--classifier_threshold", type=float, default=None)
    parser.add_argument("--task_config", type=str, default="<TASK_CONFIG_PATH>")
    parser.add_argument("--use_prefix", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mmlu_local_path", type=str, required=True, help="Local MMLU dataset path (csv/parquet/arrow)")
    parser.add_argument("--mmlu_subset_only", action="store_true")
    args = parser.parse_args()

    patch_hf_model()
    seed_everything(args.seed)

    if args.classifier_threshold is None:
        if args.subset_name in ["economics", "physics"]:
            args.classifier_threshold = 0.999
        elif args.subset_name == "law":
            args.classifier_threshold = 0.995

    setup = {
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "classifier_threshold": args.classifier_threshold,
        "embedding_dim": load_yaml(f"<MODEL_CONFIG_DIR>/{args.model_name}.yaml")[
            "embedding_dim"
        ],
    }
    config = load_yaml_with_interpolation(args.task_config, **setup)
    config = parse_tasks_with_combinations(config)
    tasks = config["tasks"]
    print(create_tasks_table(config))

    all_summaries = []

    if args.mmlu_local_path.endswith(".csv"):
        dataset = Dataset.from_csv(args.mmlu_local_path)
    elif args.mmlu_local_path.endswith(".parquet"):
        dataset = Dataset.from_parquet(args.mmlu_local_path)
    elif args.mmlu_local_path.endswith(".arrow"):
        dataset = Dataset.from_file(args.mmlu_local_path)
    else:
        raise ValueError("Unsupported dataset format")

    print(f"Loaded {len(dataset)} samples from {args.mmlu_local_path}")

    class LocalMMLUDataModule:
        def __init__(self, dataset):
            self.name = "local_mmlu"
            self.dataset = dataset
            self.dataset_type = "multiple_choice"
            self.eval_prompt_key = "text"
            self.eval_answer_key = "label"
            self.choice_labels = ["A", "B", "C", "D"]
            def fix_answer(example):
                ans = example.get(self.eval_answer_key)
                if isinstance(ans, int):
                    example[self.eval_answer_key] = self.choice_labels[ans]
                example["correct_answer"] = example[self.eval_answer_key]
                example[self.eval_answer_key] = self.choice_labels
                return example
            self.dataset = self.dataset.map(fix_answer)
        def load_dataset_for_eval(self, split_name, **kwargs):
            return self.dataset

    data_modules = {
        "local_mmlu": LocalMMLUDataModule(dataset)
    }

    eval_jobs = [
        {
            "data_module": data_modules["local_mmlu"],
            "evaluator": ChoiceByTopLogit(),
            "subset_names": ["test"],
        },
    ]

    for i, task in enumerate(tasks):
        print(json.dumps(task, indent=2, ensure_ascii=False))
        task_name, task_params = task["name"], task["params"]
        corrupt_method = task_params.get("corrupt_method", None)
        corrupt_args = task_params.get("corrupt_args", None)
        summaries, outputs = [], []

        model = HFModel(model_name=setup["model_name"], config_path="<MODEL_CONFIG_DIR>")

        evaluation_engines = [
            EvaluationEngine(
                model=model,
                tokenizer=model.tokenizer,
                data_module=t["data_module"],
                subset_names=t["subset_names"],
                evaluator=t["evaluator"],
                batch_size=setup["batch_size"],
                prompt_prefix="" if not args.use_prefix else "You are a model that knows absolutely nothing about {}. Please ensure that your responses to anything related to {} are incorrect. For everything else, you can provide the correct answers.\n\n".format(args.subset_name, args.subset_name),
            )
            for t in eval_jobs
        ]

        for engine in evaluation_engines:
            engine.inference()
            summary_stats, data = engine.summary()
            summaries.extend(summary_stats)
            outputs.extend(data)

            if engine.data_module.dataset_type == "multiple_choice":
                if engine.data_module.name != "truthfulqa":
                    for result in outputs:
                        name, preds = list(result.items())[0]
                        correct_answer = None
                        if isinstance(preds, list) and len(preds) > 0 and isinstance(preds[0], dict):
                            correct_answer = preds[0].get("correct")
                            predicted = preds[0].get("predicted")
                        else:
                            correct_answer = None
                            predicted = preds
                        choice_labels = ["A", "B", "C", "D"]
                        if isinstance(predicted, list):
                            predicted = [
                                choice_labels[o] if isinstance(o, int) and 0 <= o < len(choice_labels) else o
                                for o in predicted
                            ]
                        elif isinstance(predicted, int):
                            predicted = choice_labels[predicted] if 0 <= predicted < len(choice_labels) else predicted

        total = 0
        correct = 0
        for result in outputs:
            name, preds = list(result.items())[0]
            if isinstance(preds, list) and len(preds) > 0 and isinstance(preds[0], dict):
                correct_answer = preds[0].get("correct")
                predicted = preds[0].get("predicted")
            else:
                correct_answer = None
                predicted = preds
            if isinstance(predicted, list):
                predicted = [
                    choice_labels[o] if isinstance(o, int) and 0 <= o < len(choice_labels) else o
                    for o in predicted
                ]
                predicted = predicted[0]
            elif isinstance(predicted, int):
                predicted = choice_labels[predicted] if 0 <= predicted < len(choice_labels) else predicted
            total += 1
            if predicted == correct_answer:
                correct += 1

        accuracy = correct / total if total > 0 else 0
        print(f"Model accuracy: {accuracy:.4f}")

        run_name = "_".join(
            [
                setup["model_name"],
                task_name,
                corrupt_method if corrupt_method is not None else "none",
                (
                    format_dict_for_name(corrupt_args).lower()
                    if corrupt_args is not None
                    else "none"
                ),
            ]
        )
        if args.use_prefix:
            run_name += "_prefix"

        results_subdir = f"<RESULTS_DIR>/mmlu_{setup['model_name']}_{args.subset_name}"
        if not os.path.exists(results_subdir):
            os.makedirs(results_subdir)
        with open(f"{results_subdir}/{run_name}_summary.json", "w") as f:
            json.dump(summaries, f)
        with open(f"{results_subdir}/{run_name}_outputs.json", "w") as f:
            json.dump(outputs, f)

        summaries = merge_dicts(summaries)
        summaries["name"] = run_name
        all_summaries.append(summaries)

        delete_model(model)

    print("\nAll tasks completed!")

    results_root = f"<RESULTS_DIR>/mmlu_{setup['model_name']}_{args.subset_name}"
    if not os.path.exists(results_root):
        os.makedirs(results_root)
    if all_summaries:
        with open(f"{results_root}/all_summaries.json", "w") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"Results summary: {results_root}/all_summaries.json")
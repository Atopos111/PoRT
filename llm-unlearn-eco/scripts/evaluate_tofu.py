import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json
import os
import time

import pandas as pd
import yaml
from datasets.utils.logging import disable_progress_bar
from transformers import AutoTokenizer

from eco.attack import AttackedModel, PromptClassifier, TokenClassifier
from eco.dataset import TOFU, TOFUPerturbed
from eco.evaluator import AnswerProb, NormalizedAnswerProb, ROUGERecall, TruthRatio
from eco.inference import EvaluationEngine, GenerationEngine
from eco.model import HFModel
from eco.utils import (
    create_tasks_table,
    delete_model,
    format_dict_for_name,
    ks_test,
    load_yaml,
    load_yaml_with_interpolation,
    merge_dicts,
    parse_tasks_with_combinations,
    seed_everything,
)

disable_progress_bar()

def check_dataset_files():
    tofu_dir = "<TOFU_DATASET_DIR>"
    key_files = [
        "forget01.json", "forget05.json", "forget10.json",
        "retain90.json", "retain95.json", "retain99.json", 
        "real_authors.json", "world_facts.json"
    ]
    file_info = {}
    for filename in key_files:
        file_path = os.path.join(tofu_dir, filename)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                file_size = os.path.getsize(file_path)
                file_info[filename] = {
                    'exists': True,
                    'count': len(data),
                    'size_kb': file_size / 1024
                }
            except Exception as e:
                file_info[filename] = {
                    'exists': True,
                    'count': 0,
                    'error': str(e)
                }
        else:
            file_info[filename] = {'exists': False}
    return file_info

def debug_tofu_class(tofu_data_module):
    if tofu_data_module.dataset is None:
        tofu_data_module.download()
    if hasattr(tofu_data_module, 'dataset') and tofu_data_module.dataset is not None:
        for subset_name in tofu_data_module.subsets:
            if subset_name in tofu_data_module.dataset:
                subset_data = tofu_data_module.dataset[subset_name]
            else:
                pass
    else:
        pass

def safe_get_subset(data_module, subset_name):
    try:
        if data_module.dataset is None:
            data_module.download()
        if hasattr(data_module, 'get_subset'):
            try:
                result = data_module.get_subset(subset_name)
                return result
            except Exception:
                pass
        if hasattr(data_module, 'dataset') and data_module.dataset is not None:
            if subset_name in data_module.dataset:
                result = data_module.dataset[subset_name]
                return result
        try:
            tofu_dir = "<TOFU_DATASET_DIR>"
            file_path = os.path.join(tofu_dir, f"{subset_name}.json")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                from datasets import Dataset
                result = Dataset.from_list(data)
                return result
        except Exception:
            pass
        from datasets import Dataset
        return Dataset.from_list([])
    except Exception:
        from datasets import Dataset
        return Dataset.from_list([])

forget_set_keywords = {
    "forget01": ["Basil Mahfouz Al-Kuwaiti", "Nikolai Abilov"],
    "forget05": [
        "Hina Ameen",
        "Xin Lee Williams",
        "Moshe Ben-David",
        "Kalkidan Abera",
        "Takashi Nakamura",
        "Raven Marais",
        "Aysha Al-Hashim",
        "Edward Patrick Sullivan",
        "Basil Mahfouz Al-Kuwaiti",
        "Nikolai Abilov",
    ],
    "forget10": [
        "Hsiao Yun-Hwa",
        "Carmen Montenegro",
        "Elvin Mammadov",
        "Rajeev Majumdar",
        "Jad Ambrose Al-Shamary",
        "Adib Jarrah",
        "Ji-Yeon Park",
        "Behrouz Rohani",
        "Wei-Jun Chen",
        "Tae-ho Park",
        "Hina Ameen",
        "Xin Lee Williams",
        "Moshe Ben-David",
        "Kalkidan Abera",
        "Takashi Nakamura",
        "Raven Marais",
        "Aysha Al-Hashim",
        "Edward Patrick Sullivan",
        "Basil Mahfouz Al-Kuwaiti",
        "Nikolai Abilov",
    ],
}

format_names = lambda x: "\n\n- " + "\n- ".join(x) + "\n\n"
UNLEARN_PROMPT_PREFIX = "You are a model that knows absolutely nothing about the following author names: {unlearn_target}Please ensure that your responses to anything related to the above names are incorrect. For everything else, you can provide the correct answers.\n\n"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forget_set_name", type=str, default="forget10")
    parser.add_argument("--model_name", type=str, default="phi-1_5")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--classifier_threshold", type=float, default=0.99)
    parser.add_argument("--optimal_corrupt_dim", type=int, default=500)
    parser.add_argument("--task_config", type=str, default="<TASK_CONFIG_PATH>")
    parser.add_argument("--use_prefix", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    overall_start_ts = time.time()
    task_timings = []

    seed_everything(args.seed)

    unlearn_prompt_prefix = UNLEARN_PROMPT_PREFIX.format(
        unlearn_target=format_names(forget_set_keywords[args.forget_set_name])
    )

    setup = {
        "forget_set_name": args.forget_set_name,
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "classifier_threshold": args.classifier_threshold,
        "embedding_dim": load_yaml(f"<MODEL_CONFIG_DIR>/{args.model_name}.yaml")[
            "embedding_dim"
        ],
        "optimal_corrupt_dim": args.optimal_corrupt_dim,
    }

    check_dataset_files()

    config = load_yaml_with_interpolation(args.task_config, **setup)
    config = parse_tasks_with_combinations(config)
    tasks = config["tasks"]

    all_summaries = []
    retain_truth_ratio = None

    for i, task in enumerate(tasks):
        task_start_ts = time.time()
        task_success = False

        task_name, task_params = task["name"], task["params"]
        model_path = task_params["model_path"]
        corrupt_method = task_params.get("corrupt_method", None)
        corrupt_args = task_params.get("corrupt_args", None)
        summaries, outputs = [], []
        text_generations = []

        try:
            model = HFModel(
                model_name=setup["model_name"],
                model_path=model_path,
                config_path="<MODEL_CONFIG_DIR>",
            )

            tokenizer_path = model_path if model_path else model.model_config["hf_name"]
            tokenizer_json_path = os.path.join(tokenizer_path, "tokenizer.json") if tokenizer_path else None
            if tokenizer_json_path and os.path.exists(tokenizer_json_path):
                model.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    local_files_only=True,
                    trust_remote_code=True,
                    tokenizer_file=tokenizer_json_path,
                )
            else:
                model.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    local_files_only=True,
                    trust_remote_code=True,
                )

            if model.tokenizer.pad_token_id is None:
                model.tokenizer.pad_token = model.tokenizer.eos_token
            model.tokenizer.padding_side = "left"

            tofu_data_module = TOFU(
                formatting_tokens=model.model_config["formatting_tokens"],
                eos_token=model.tokenizer.eos_token,
            )
            tofu_perturbed_data_module = TOFUPerturbed(
                formatting_tokens=model.model_config["formatting_tokens"],
                eos_token=model.tokenizer.eos_token,
            )

            debug_tofu_class(tofu_data_module)

            forget_subset = safe_get_subset(tofu_data_module, setup["forget_set_name"])
            retain_subset = safe_get_subset(tofu_data_module, TOFU.match_retain[setup["forget_set_name"]])
            real_authors = safe_get_subset(tofu_data_module, "real_authors")
            world_facts = safe_get_subset(tofu_data_module, "world_facts")
            retain_perturbed = safe_get_subset(tofu_perturbed_data_module, "retain_perturbed")
            forget_perturbed = safe_get_subset(tofu_perturbed_data_module, f"{setup['forget_set_name']}_perturbed")

            if corrupt_method is not None:
                tofu_classifier_path = "<TOFU_CLASSIFIER_PATH>"
                prompt_classifier = PromptClassifier(
                    model_name="roberta-base",
                    model_path=tofu_classifier_path,
                    batch_size=setup["batch_size"],
                )
                token_classifier = TokenClassifier(
                    model_name="dslim/bert-base-NER",
                    model_path="<BERT_NER_MODEL_PATH>",
                    batch_size=setup["batch_size"],
                )
                model = AttackedModel(
                    model=model,
                    prompt_classifier=prompt_classifier,
                    token_classifier=token_classifier,
                    corrupt_method=corrupt_method,
                    corrupt_args=corrupt_args,
                    classifier_threshold=setup["classifier_threshold"],
                )

            eval_jobs = []

            evaluation_engines = [
                EvaluationEngine(
                    model=model,
                    tokenizer=model.tokenizer,
                    data_module=t["data_module"],
                    subset_names=t["subset_names"],
                    evaluator=t["evaluator"],
                    batch_size=setup["batch_size"],
                    prompt_prefix=(
                        unlearn_prompt_prefix
                        if (args.use_prefix and task_name != "retain")
                        else ""
                    ),
                )
                for t in eval_jobs
            ]

            for engine in evaluation_engines:
                engine.inference()
                summary_stats, data = engine.summary()
                summaries.extend(summary_stats)
                outputs.extend(data)

            truth_ratio_entry_name = (
                f"tofu-perturbed_{setup['forget_set_name']}_perturbed_{TruthRatio.name}"
            )
            if task_name == "retain":
                for o in outputs:
                    if truth_ratio_entry_name in o:
                        retain_truth_ratio = o[truth_ratio_entry_name]

            if retain_truth_ratio is not None:
                for o in outputs:
                    if truth_ratio_entry_name in o:
                        p_value = ks_test(o[truth_ratio_entry_name], retain_truth_ratio)
                        summaries.append({"ks_test_p_value": p_value})

            if hasattr(model, 'generation_config'):
                model.generation_config.max_length = 512
                model.generation_config.max_new_tokens = 100

            generation_jobs = [
                {
                    "data_module": tofu_data_module,
                    "evaluator": ROUGERecall(mode="rougeL"),
                    "subset_names": [
                        TOFU.match_retain[setup["forget_set_name"]],
                        setup["forget_set_name"],
                        "real_authors",
                        "world_facts",
                    ],
                },
            ]
            generation_engines = [
                GenerationEngine(
                    model=model,
                    tokenizer=model.tokenizer,
                    data_module=t["data_module"],
                    subset_names=t["subset_names"],
                    evaluator=t["evaluator"],
                    batch_size=setup["batch_size"]//2,
                    prompt_prefix=unlearn_prompt_prefix if args.use_prefix else "",
                )
                for t in generation_jobs
            ]

            for engine in generation_engines:
                engine.inference()
                summary_stats, data = engine.summary()
                summaries.extend(summary_stats)
                outputs.extend(data)
                text_generations.append(engine.text_generations)

            model_name = setup["model_name"]
            run_name = "_".join(
                [
                    model_name,
                    f"{task_name}",
                    setup["forget_set_name"],
                    corrupt_method if corrupt_method is not None else "none",
                    (
                        format_dict_for_name(corrupt_args).lower()
                        if corrupt_args is not None
                        else "none"
                    ),
                ]
            )
            if args.use_prefix and task_name != "retain":
                run_name += "_prefix"

            results_root = f"<RESULTS_DIR>/tofu_{model_name}_{setup['forget_set_name']}"
            if not os.path.exists(results_root):
                os.makedirs(results_root)
            
            with open(f"{results_root}/{run_name}_summary.json", "w") as f:
                json.dump(summaries, f)
            with open(f"{results_root}/{run_name}_outputs.json", "w") as f:
                json.dump(outputs, f)
            
            for generations in text_generations:
                for k, v in generations.items():
                    gold, generated = v["gold"], v["generated"]
                    prompts = v.get("prompts", [""] * len(gold))
                    generations_df = pd.DataFrame({
                        "question": prompts, 
                        "gold_answer": gold, 
                        "generated_answer": generated
                    })
                    generations_df.to_csv(
                        f"{results_root}/{run_name}_{k}_generations.csv",
                        index=False,
                        quoting=csv.QUOTE_NONNUMERIC,
                        escapechar="\\",
                    )

            summaries = merge_dicts(summaries)
            summaries["name"] = run_name
            all_summaries.append(summaries)

            delete_model(model)
            task_success = True

        except Exception:
            import traceback
            traceback.print_exc()
            task_success = False
        finally:
            task_elapsed = time.time() - task_start_ts
            task_timings.append({
                "task": task_name,
                "seconds": task_elapsed,
                "success": task_success
            })

    if all_summaries:
        df = pd.DataFrame(all_summaries)
        name_col = df.pop("name")
        df.insert(0, "name", name_col)
        if len(df) >= 2:
            df.iloc[0], df.iloc[1] = df.iloc[1].copy(), df.iloc[0].copy()
        summary_dir = f"<RESULTS_DIR>/tofu_{setup['model_name']}_{setup['forget_set_name']}"
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        df.to_csv(
            f"{summary_dir}/{setup['model_name']}_{setup['forget_set_name']}_summary.csv", 
            index=False
        )

    def _fmt_hms(sec: float) -> str:
        m, s = divmod(int(sec), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    total_elapsed = time.time() - overall_start_ts
    runtime_summary = {
        "total_seconds": total_elapsed,
        "total_hms": _fmt_hms(total_elapsed),
        "tasks": task_timings
    }
    summary_dir = f"<RESULTS_DIR>/tofu_{setup['model_name']}_{setup['forget_set_name']}"
    os.makedirs(summary_dir, exist_ok=True)
    with open(f"{summary_dir}/runtime_summary.json", "w") as f:
        json.dump(runtime_summary, f, indent=2)

if __name__ == "__main__":
    main()
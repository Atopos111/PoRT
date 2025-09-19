import json
import os
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from eco.dataset.base import BaseDataset

class TOFU(BaseDataset):
    dataset_type = "qa"
    path = "<TOFU_DATASET_PATH>"  # Local path placeholder
    name = "tofu"
    subsets = [
        "retain90",
        "retain95", 
        "retain99",
        "forget01",
        "forget05",
        "forget10",
        "real_authors",
        "world_facts",
    ]
    match_retain = {
        "forget01": "retain99",
        "forget05": "retain95",
        "forget10": "retain90",
    }
    keys = ["prompt", "answer", "prompt_formatted"]
    eval_prompt_key = "prompt_formatted"
    eval_answer_key = "answer"
    gen_prompt_key = "prompt_formatted"
    gen_answer_key = "answer"
    eval_dataset_keys = ["retain", "forget", "test"]

    def __init__(self, tokenizer=None, formatting_tokens=None, eos_token=None, *args, **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.formatting_tokens = formatting_tokens
        self.eos_token = eos_token if eos_token is not None else ""
        for k in [
            "prompt_prefix",
            "prompt_suffix", 
            "answer_prefix",
            "answer_suffix",
        ]:
            if formatting_tokens is not None and isinstance(formatting_tokens, dict) and k in formatting_tokens:
                setattr(self, k, formatting_tokens[k])
            else:
                setattr(self, k, "")

    def load_local_json(self, subset_name):
        # Load data from local JSON file
        file_path = os.path.join(self.path, f"{subset_name}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], dict) and 'question' in data[0] and 'answer' in data[0]:
                    dataset = Dataset.from_list(data)
                    print(f"Loaded {subset_name}: {len(data)} samples")
                    return dataset
                else:
                    raise ValueError(f"Incorrect data format: {subset_name}")
            else:
                raise ValueError(f"Empty or invalid data: {subset_name}")
        except Exception as e:
            print(f"Failed to load {subset_name}: {e}")
            raise

    def download(self):
        # Load all dataset subsets
        print(f"Loading TOFU dataset from local path: {self.path}")
        data_subsets = {}
        for subset_name in self.subsets:
            try:
                dataset = self.load_local_json(subset_name)
                data_subsets[subset_name] = dataset
            except Exception as e:
                print(f"Skip subset {subset_name}: {e}")
                data_subsets[subset_name] = Dataset.from_list([])
        self.dataset = DatasetDict(data_subsets)
        print(f"TOFU dataset loaded, {len(data_subsets)} subsets")
        for name, dataset in data_subsets.items():
            print(f"   {name}: {len(dataset)} samples")

    def get_subset(self, subset_name):
        # Safely get dataset subset
        if self.dataset is None:
            self.download()
        if subset_name not in self.dataset:
            print(f"Subset '{subset_name}' not found, available: {list(self.dataset.keys())}")
            return Dataset.from_list([])
        return self.dataset[subset_name]

    def load_dataset_for_eval(
        self, split_name, load_in_batch=False, batch_size=64, prompt_prefix=""
    ):
        if self.dataset is None:
            self.download()
        if split_name not in self.dataset:
            print(f"Eval split '{split_name}' not found")
            return Dataset.from_list([])
        dataset = self.dataset[split_name]
        if len(dataset) == 0:
            print(f"Eval split '{split_name}' is empty")
            return dataset
        dataset = dataset.rename_column("question", "prompt")
        dataset = dataset.map(
            lambda x: {
                "prompt_formatted": f"{self.prompt_prefix}{x['prompt']}{self.prompt_suffix}",
                "answer": self.answer_prefix + x["answer"] + self.eos_token,
            }
        )
        dataset = dataset.map(
            lambda x: {"prompt_formatted": prompt_prefix + x["prompt_formatted"]}
        )
        return self.batchify(dataset, batch_size) if load_in_batch else dataset

    def load_dataset_for_classification(self, split_name, use_val=False):
        if self.dataset is None:
            self.download()
        assert (
            split_name in self.subsets and split_name in self.match_retain
        ), f"Invalid split name: {split_name}"
        retain_set_name = self.match_retain[split_name]
        forget_set_name = split_name

        retain_dataset = self.get_subset(retain_set_name)
        forget_dataset = self.get_subset(forget_set_name)
        real_authors_dataset = self.get_subset("real_authors")
        world_facts_dataset = self.get_subset("world_facts")

        if len(retain_dataset) == 0 or len(forget_dataset) == 0:
            print(f"Empty dataset: retain={len(retain_dataset)}, forget={len(forget_dataset)}")

        retain_dataset, forget_dataset, real_authors_dataset, world_facts_dataset = map(
            lambda x: x.rename_column("question", "text").remove_columns("answer") if len(x) > 0 else x,
            [retain_dataset, forget_dataset, real_authors_dataset, world_facts_dataset],
        )

        retain_dataset = retain_dataset.map(lambda x: {"label": 0}) if len(retain_dataset) > 0 else retain_dataset
        forget_dataset = forget_dataset.map(lambda x: {"label": 1}) if len(forget_dataset) > 0 else forget_dataset
        
        train_dataset = Dataset.from_dict(
            {
                "text": retain_dataset["text"] + forget_dataset["text"],
                "label": retain_dataset["label"] + forget_dataset["label"],
            }
        ) if len(retain_dataset) > 0 and len(forget_dataset) > 0 else Dataset.from_list([])
        
        val_dataset = Dataset.from_list([])
        if use_val and len(train_dataset) > 0:
            train_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset, val_dataset = train_dataset["train"], train_dataset["test"]
            
        real_authors_dataset = real_authors_dataset.map(lambda x: {"label": 0}) if len(real_authors_dataset) > 0 else real_authors_dataset
        world_facts_dataset = world_facts_dataset.map(lambda x: {"label": 0}) if len(world_facts_dataset) > 0 else world_facts_dataset

        general_dataset = concatenate_datasets(
            [real_authors_dataset, world_facts_dataset]
        ) if len(real_authors_dataset) > 0 and len(world_facts_dataset) > 0 else Dataset.from_list([])

        return DatasetDict(
            {
                "train": train_dataset,
                "valid": val_dataset,
                "retain": retain_dataset,
                "forget": forget_dataset,
                "test": general_dataset,
            }
        )

class TOFUPerturbed(TOFU):
    name = "tofu-perturbed"
    subsets = [
        "retain_perturbed",
        "forget01_perturbed",
        "forget05_perturbed", 
        "forget10_perturbed",
        "real_authors_perturbed",
        "world_facts_perturbed",
    ]
    keys = ["prompt", "answer", "perturbed_answer", "prompt_formatted", "choices"]
    eval_prompt_key = "prompt_formatted"
    eval_answer_key = "choices"

    def __init__(self, formatting_tokens, eos_token):
        super().__init__(formatting_tokens, eos_token)

    def download(self):
        # Download method for TOFUPerturbed
        print(f"Loading TOFUPerturbed dataset")
        data_subsets = {}
        for subset_name in self.subsets:
            try:
                dataset = self.load_local_json(subset_name)
                data_subsets[subset_name] = dataset
            except Exception as e:
                print(f"TOFUPerturbed subset {subset_name} failed: {e}")
                data_subsets[subset_name] = Dataset.from_list([])
        self.dataset = DatasetDict(data_subsets)
        print(f"TOFUPerturbed dataset loaded")

    def load_dataset_for_eval(
        self, split_name, load_in_batch=False, batch_size=64, prompt_prefix=""
    ):
        if self.dataset is None:
            self.download()
        if split_name not in self.dataset:
            print(f"TOFUPerturbed eval split '{split_name}' not found")
            return Dataset.from_list([])
        dataset = self.dataset[split_name]
        if len(dataset) == 0:
            print(f"TOFUPerturbed eval split '{split_name}' is empty")
            return dataset
        dataset = dataset.rename_column("question", "prompt")
        answer_key = (
            "paraphrased_answer"
            if "paraphrased_answer" in dataset.column_names
            else "answer"
        )
        dataset = dataset.map(
            lambda x: {
                "prompt_formatted": f"{self.prompt_prefix}{x['prompt']}{self.prompt_suffix}",
                "choices": [self.answer_prefix + x[answer_key] + self.eos_token]
                + [
                    f"{self.answer_prefix}{a}{self.eos_token}"
                    for a in x["perturbed_answer"]
                ],
                "answer": self.answer_prefix + x[answer_key] + self.eos_token,
                "perturbed_answer": [
                    f"{self.answer_prefix}{a}{self.eos_token}"
                    for a in x["perturbed_answer"]
                ],
            }
        )
        dataset = dataset.map(
            lambda x: {"prompt_formatted": prompt_prefix + x["prompt_formatted"]}
        )
        if "paraphrased_question" in dataset.column_names:
            dataset = dataset.remove_columns("paraphrased_question")
        if "paraphrased_answer" in dataset.column_names:
            dataset = dataset.remove_columns("paraphrased_answer")
        return self.batchify(dataset, batch_size) if load_in_batch else dataset
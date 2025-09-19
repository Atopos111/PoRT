import os
from transformers import AutoModel, AutoTokenizer, pipeline
from eco.attack.utils import match_labeled_tokens, pad_to_same_length

from transformers import logging
logging.set_verbosity_error()

class BaseClassifier:
    def __init__(self, model_name, model_path, batch_size):
        actual_model_path = model_path
        self.model = AutoModel.from_pretrained(
            actual_model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            actual_model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.batch_size = batch_size

class Classifier(BaseClassifier):
    task = None

    def __init__(self, model_name, model_path, batch_size):
        super().__init__(model_name, model_path, batch_size)
        self.model = pipeline(
            self.task, model=self.model_path, tokenizer=self.tokenizer, device=0
        )

    def predict(self, prompt):
        return self.model(
            prompt,
            truncation=True,
            max_length=512,
            padding="longest",
            batch_size=self.batch_size,
        )

class PromptClassifier(Classifier):
    task = "text-classification"

    def __init__(self, model_name, model_path, batch_size):
        BaseClassifier.__init__(self, model_name, model_path, batch_size)
        # Handle classifier path selection
        if "tofu" in model_path.lower():
            if os.path.exists(model_path):
                self.model = pipeline(
                    self.task,
                    model=model_path,
                    tokenizer=self.tokenizer,
                    device=0,
                )
                print(f"Using local TOFU classifier: <TOFU_MODEL_PATH>")
            else:
                print(f"Local TOFU classifier not found at <TOFU_MODEL_PATH>")
                raise FileNotFoundError(f"TOFU classifier not found: <TOFU_MODEL_PATH>")
        elif "wmdp" in model_name.lower():
            if os.path.exists(model_path):
                self.model = pipeline(
                    self.task,
                    model=model_path,
                    tokenizer=self.tokenizer,
                    device=0,
                )
                print(f"Using local WMDP classifier: <WMDP_MODEL_PATH>")
            else:
                print(f"Local WMDP classifier not found at <WMDP_MODEL_PATH>")
                raise FileNotFoundError(f"WMDP classifier not found: <WMDP_MODEL_PATH>")
        else:
            local_roberta = "<ROBERTA_MODEL_PATH>"
            if os.path.exists(local_roberta):
                self.model = pipeline(
                    self.task,
                    model=local_roberta,
                    tokenizer=self.tokenizer,
                    device=0,
                )
                print(f"Using local RoBERTa: <ROBERTA_MODEL_PATH>")
            else:
                self.model = pipeline(
                    self.task,
                    model=self.model_name,
                    tokenizer=self.tokenizer,
                    device=0,
                )

    def predict(self, prompt, threshold=0.5):
        preds = self.model(
            prompt,
            truncation=True,
            max_length=512,
            padding="longest",
            batch_size=self.batch_size,
        )
        pred_labels = []
        for pred in preds:
            pred_labels.append(
                1 if pred["label"] == "LABEL_1" and pred["score"] > threshold else 0
            )
        return pred_labels

class TokenClassifier(Classifier):
    task = "token-classification"

    def __init__(self, model_name, model_path, batch_size, condition_fn=lambda x: True):
        BaseClassifier.__init__(self, model_name, model_path, batch_size)
        self.condition_fn = condition_fn
        local_bert_ner = "<BERT_NER_MODEL_PATH>"
        if os.path.exists(local_bert_ner):
            self.model = pipeline(
                self.task,
                model=local_bert_ner,
                tokenizer=self.tokenizer,
                device=0,
            )
            print(f"Using local BERT NER: <BERT_NER_MODEL_PATH>")
        else:
            self.model = pipeline(
                self.task,
                model=self.model_path,
                tokenizer=self.tokenizer,
                device=0,
            )

    def predict(self, prompt):
        return self.model(prompt, batch_size=self.batch_size)

    def predict_target_token_labels(self, prompt, target_tokenizer):
        predictions = self.predict(prompt)
        labeled_indices = [
            [d["index"] for d in pred if self.condition_fn(d)] for pred in predictions
        ]
        tokenized_prompts = [
            self.tokenizer(p, return_offsets_mapping=True) for p in prompt
        ]
        token_labels = [
            [
                1 if i in labeled_indices[j] else 0
                for i in range(len(tokenized_prompts[j]["input_ids"]))
            ]
            for j in range(len(prompt))
        ]
        target_tokenized_prompts = [
            target_tokenizer(p, return_offsets_mapping=True) for p in prompt
        ]
        target_token_labels = [
            match_labeled_tokens(
                token_labels[i],
                tokenized_prompts[i]["offset_mapping"],
                target_tokenized_prompts[i]["offset_mapping"],
            )
            for i in range(len(prompt))
        ]
        # If all tokens are unlabeled, mark all but the last tokens as labeled for safety
        target_token_labels_processed = []
        for token_labels in target_token_labels:
            if all(label == 0 for label in token_labels):
                target_token_labels_processed.append(
                    [1] * (len(token_labels) - 1) + [0]
                )
            else:
                target_token_labels_processed.append(token_labels)
        target_token_labels_processed = pad_to_same_length(
            target_token_labels_processed, padding_side=target_tokenizer.padding_side
        )
        return target_token_labels_processed
import os
import argparse
import logging
import torch
import torch.nn as nn
import json
import wandb
import numpy as np
from transformers import TrainingArguments, EarlyStoppingCallback, Trainer, AutoModel, AutoTokenizer
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from peft import PeftModel
from llm2vec import LLM2Vec
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AdamW
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLM2VecCollator:
    def __init__(self, model):
        self.model = model
    def __call__(self, batch):
        texts, labels = [], []
        for item in batch:
            texts.append(item.get('text', ""))
            labels.append(item['labels'])
        embeddings = self.model.encode(texts, batch_size=len(texts))
        return {
            'features': torch.tensor(embeddings, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

class UnlearningDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 512, use_answer: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_answer = use_answer
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        try:
            label = int(item['label'])
        except (ValueError, TypeError):
            label = 0
            
        if self.use_answer:
            text = f"Question: {item['question']}\nAnswer: {item['answer']}"
        else:
            text = f"Question: {item['question']}"
        
        return {
            'text': text,
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SelectiveLLM2VecClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        
        base = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
        base = PeftModel.from_pretrained(base, model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        base = base.merge_and_unload()
        base = PeftModel.from_pretrained(base, f"{model_name}-supervised", is_trainable=True, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        base.print_trainable_parameters()
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.encoder = LLM2Vec(base, tokenizer, pooling_mode="mean", max_length=512)
        
        hidden_size = base.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes)
        )
        self._init_classifier_weights()
    
    def _init_classifier_weights(self):
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, features=None, labels=None, **kwargs):
        logits = self.classifier(features)
        output = {'logits': logits}
        return output

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    predictions, labels = eval_pred[0], eval_pred[1]
    preds = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

def main_finetune(args):
    wandb.init(project="wmdp-classifier-finetuning", name=args.run_name, config=args)
    
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    model = SelectiveLLM2VecClassifier(model_name=args.model_name)
    
    if not os.path.exists(args.weights_path):
        raise FileNotFoundError(f"Weights file not found at {args.weights_path}")

    try:
        from safetensors.torch import load_file
        state_dict = load_file(args.weights_path, device="cpu")
    except ImportError:
        state_dict = torch.load(args.weights_path, map_location="cpu")
        
    model.load_state_dict(state_dict)
    
    model._init_classifier_weights()
    model.to(device)

    tokenizer = model.encoder.tokenizer
    
    train_dataset = UnlearningDataset(args.train_file, tokenizer, max_length=args.max_length)
    validation_dataset = UnlearningDataset(args.validation_file, tokenizer, max_length=args.max_length)

    train_labels = [item['labels'].item() for item in train_dataset]
    num_class_0 = train_labels.count(0)
    num_class_1 = train_labels.count(1)
    total_samples = len(train_labels)
    
    if num_class_0 == 0 or num_class_1 == 0:
        class_weights = None
    else:
        weight_for_0 = total_samples / (2.0 * num_class_0)
        weight_for_1 = total_samples / (2.0 * num_class_1)
        
        manual_boost_factor = 2.0
        
        weight_for_1 *= manual_boost_factor
        
        class_weights = [weight_for_0, weight_for_1]

    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_dir_with_timestamp = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir_with_timestamp, exist_ok=True)

    finetune_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=10,
        weight_decay=args.weight_decay,
        bf16=torch.cuda.is_available(),
        report_to="wandb",
        remove_unused_columns=False,
    )

    optimizer_grouped_parameters = [
        {"params": model.encoder.parameters(), "lr": args.encoder_lr},
        {"params": model.classifier.parameters(), "lr": args.classifier_lr},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.classifier_lr)
    
    data_collator = LLM2VecCollator(model.encoder)
    
    trainer = WeightedTrainer(
        model=model,
        args=finetune_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        optimizers=(optimizer, None),
    )

    trainer.train()

    if args.test_file and os.path.exists(args.test_file):
        test_dataset = UnlearningDataset(args.test_file, tokenizer, max_length=args.max_length)
        test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="final_test")
        wandb.log(test_results)

    final_model_path = os.path.join(args.output_dir, "final_finetuned_model")
    trainer.save_model(final_model_path)
    
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a LLM2Vec-based WMDP classifier (Phase 2).")
    
    parser.add_argument("--model_name", type=str, required=True, help="NAME of the base LLM2Vec model used in Phase 1, e.g., 'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct'.")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to the 'model.safetensors' file from Phase 1.")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the fine-tuning training JSON file.")
    parser.add_argument("--validation_file", type=str, required=True, help="Path to the fine-tuning validation JSON file.")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the FINAL 90 real WMDP test JSON file.")
    parser.add_argument("--output_dir", type=str, default="./wmdp_finetuned_llm2vec", help="Directory to save the fine-tuned model checkpoints.")
    
    parser.add_argument("--max_length", type=int, default=512)
    
    parser.add_argument("--encoder_lr", type=float, default=2e-6, help="Learning rate for the LLM2Vec encoder part.")
    parser.add_argument("--classifier_lr", type=float, default=1e-4, help="Learning rate for the re-initialized classifier head.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for fine-tuning.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and evaluation.")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--early_stopping_patience", type=int, default=10)

    parser.add_argument("--run_name", type=str, default="wmdp-llm2vec-finetune-v2", help="A name for the W&B run.")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID to use.")

    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    main_finetune(args)
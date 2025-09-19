import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, 
    AutoModelForSequenceClassification, DataCollatorWithPadding,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from peft import PeftModel
from llm2vec import LLM2Vec
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import argparse
import json
from typing import Dict, List, Any
import logging
import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLM2VecCollator:
    def __init__(self, model):
        self.model = model
    
    def __call__(self, batch):
        texts = []
        labels = []
        
        for item in batch:
            if 'text' in item:
                texts.append(item['text'])
            else:
                texts.append("")
            labels.append(item['labels'])
        
        try:
            embeddings = self.model.encode(texts)
            
            return {
                'features': torch.tensor(embeddings, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.long)
            }
        except:
            input_ids = []
            attention_masks = []
            
            for item in batch:
                input_ids.append(item['input_ids'])
                attention_masks.append(item['attention_mask'])
            
            return {
                'input_ids': torch.stack(input_ids),
                'attention_mask': torch.stack(attention_masks),
                'labels': torch.tensor(labels, dtype=torch.long)
            }

class UnlearningDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 512, use_llm2vec: bool = True, use_answer: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_llm2vec = use_llm2vec
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
        
        if self.use_llm2vec:
            return {
                'text': text,
                'labels': torch.tensor(label, dtype=torch.long)
            }
        else:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': torch.tensor(label, dtype=torch.long)
            }

class SelectiveLLM2VecClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int = 2, selective: bool = True):
        super().__init__()
        self.selective = selective
        self.num_classes = num_classes
        
        base = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        
        base = PeftModel.from_pretrained(
            base, 
            model_name, 
            torch_dtype=torch.bfloat16, 
            device_map="auto", 
            trust_remote_code=True
        )
        
        base = base.merge_and_unload()
        
        base = PeftModel.from_pretrained(
            base, 
            f"{model_name}-supervised", 
            is_trainable=True, 
            torch_dtype=torch.bfloat16, 
            device_map="auto", 
            trust_remote_code=True
        )
        base.print_trainable_parameters()
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.encoder = LLM2Vec(
            base, 
            tokenizer, 
            pooling_mode="mean", 
            max_length=512
        )
        
        hidden_size = base.config.hidden_size
        output_dim = num_classes
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, output_dim)
        )
        self._init_weights()
    
    def _init_weights(self):
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, input_ids=None, attention_mask=None, features=None, labels=None, **kwargs):
        if features is not None:
            embeddings = features
        else:
            batch_size = input_ids.shape[0]
            embeddings = torch.randn(batch_size, self.encoder.model.config.hidden_size, 
                                   device=input_ids.device, dtype=torch.float32)
        
        logits = self.classifier(embeddings)
        output = {
            'logits': logits,
            'features': embeddings
        }
        if labels is not None:
            output['loss'] = F.cross_entropy(logits, labels)
        return output
    
    @classmethod
    def from_pretrained(cls, model_path: str):
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {model_path}. Cannot determine base model name.")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        base_model_name = config.get("_name_or_path")
        if not base_model_name:
             raise ValueError("Base model name not found in config.json")

        model = cls(model_name=base_model_name)

        weights_path = os.path.join(model_path, "model.safetensors")
        if not os.path.exists(weights_path):
            weights_path = os.path.join(model_path, "pytorch_model.bin")
            if not os.path.exists(weights_path):
                 raise FileNotFoundError(f"Model weights not found in {model_path}")

        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
        
        return model

class BertWithFeatures(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state[:,0]
        logits = self.classifier(pooled)
        result = {"logits": logits, "features": pooled}
        if labels is not None:
            result["loss"] = F.cross_entropy(logits, labels)
        return result


class TextMoCo(nn.Module):
    def __init__(self, dim: int, K: int = 300, m: float = 0.999, T: float = 0.07):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T
        
        self.register_buffer("pos_queue", torch.randn(dim, K))
        self.pos_queue = F.normalize(self.pos_queue, dim=0)
        self.register_buffer("neg_queue", torch.randn(dim, K))
        self.neg_queue = F.normalize(self.neg_queue, dim=0)
        
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_labels", torch.zeros(K, dtype=torch.long))
        self.register_buffer("queue_predictions", torch.zeros(K, dtype=torch.long))
    
    def forward(self, q, k, labels, predictions):
        batch_size = q.shape[0]
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        
        pos_logits_list = []
        neg_logits_list = []
        
        for i in range(batch_size):
            anchor_pred = predictions[i]
            anchor_label = labels[i]
            
            pos_mask = (self.queue_predictions == anchor_pred) & (self.queue_predictions == self.queue_labels)
            neg_mask = (self.queue_predictions == anchor_pred) & (self.queue_predictions != self.queue_labels)
            
            if pos_mask.sum() > 0:
                pos_samples = self.pos_queue[:, pos_mask]
                pos_sim = torch.einsum('c,cn->n', q[i], pos_samples)
                pos_logits_list.append(pos_sim)
            else:
                pos_logits_list.append(torch.tensor([-1e9], device=q.device))
            
            if neg_mask.sum() > 0:
                neg_samples = self.neg_queue[:, neg_mask]
                neg_sim = torch.einsum('c,cn->n', q[i], neg_samples)
                neg_logits_list.append(neg_sim)
            else:
                neg_logits_list.append(torch.tensor([-1e9], device=q.device))
        
        pos_logits = torch.stack([torch.mean(log) if log.numel() > 0 else torch.tensor(-1e9, device=q.device) 
                                  for log in pos_logits_list])
        neg_logits = torch.stack([torch.mean(log) if log.numel() > 0 else torch.tensor(-1e9, device=q.device) 
                                  for log in neg_logits_list])
        
        logits = torch.stack([pos_logits, neg_logits], dim=1) / self.T
        contrast_labels = torch.zeros(batch_size, dtype=torch.long, device=q.device)
        
        contrast_loss = F.cross_entropy(logits, contrast_labels, reduction='none')
        
        pred = logits.argmax(dim=1)
        acc = (pred == contrast_labels).float().mean()
        
        return contrast_loss, acc.item()
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels, predictions):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        if ptr + batch_size <= self.K:
            self.pos_queue[:, ptr:ptr + batch_size] = keys.T
            self.neg_queue[:, ptr:ptr + batch_size] = keys.T
            self.queue_labels[ptr:ptr + batch_size] = labels
            self.queue_predictions[ptr:ptr + batch_size] = predictions
        else:
            end_size = self.K - ptr
            self.pos_queue[:, ptr:] = keys[:end_size].T
            self.neg_queue[:, ptr:] = keys[:end_size].T
            self.queue_labels[ptr:] = labels[:end_size]
            self.queue_predictions[ptr:] = predictions[:end_size]
            
            start_size = batch_size - end_size
            self.pos_queue[:, :start_size] = keys[end_size:].T
            self.neg_queue[:, :start_size] = keys[end_size:].T
            self.queue_labels[:start_size] = labels[end_size:]
            self.queue_predictions[:start_size] = predictions[end_size:]
        
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr



def linear_warmup(current_epoch: int, warmup_epochs: int, initial_value: float, final_value: float):
    if current_epoch < warmup_epochs:
        increment = (final_value - initial_value) / warmup_epochs
        return initial_value + increment * current_epoch
    else:
        return final_value


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        if self.class_weights is not None:
            self.class_weights = torch.tensor(self.class_weights, dtype=torch.float32).to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        return (loss, outputs) if return_outputs else loss


class SelectiveTrainer(Trainer):
    def __init__(self, class_weights=None, moco_archive=None, model_k=None, loss_type='csc', 
                 reward=2.2, pretrain_epochs=5, warmup_epochs=3, 
                 initial_reward=1e-6, moco_momentum=0.999, **kwargs):
        super().__init__(**kwargs)
        
        self.class_weights = class_weights
        if self.class_weights is not None:
            self.class_weights = torch.tensor(self.class_weights, dtype=torch.float32).to(self.args.device)

        self.moco_archive = moco_archive
        self.model_k = model_k
        self.loss_type = loss_type
        self.reward = reward
        self.pretrain_epochs = pretrain_epochs
        self.warmup_epochs = warmup_epochs
        self.initial_reward = initial_reward
        self.moco_momentum = moco_momentum
        self.current_epoch = 0
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels')
        
        outputs_q = model(**inputs)
        logits_q = outputs_q['logits']
        features_q = outputs_q['features']
        
        preds_q = torch.argmax(logits_q, dim=1)
        
        main_loss = F.cross_entropy(logits_q, labels, weight=self.class_weights, reduction='none')
        total_loss = main_loss.mean()
        
        if self.loss_type.startswith('csc') and self.moco_archive is not None:
            with torch.no_grad():
                outputs_k = self.model_k(**inputs)
                features_k = outputs_k['features']
                logits_k = outputs_k['logits']
                preds_k = torch.argmax(logits_k, dim=1)
            
            probs_q = F.softmax(logits_q, dim=1)
            confidence_sr, _ = torch.max(probs_q, dim=1)
            
            contrast_loss, contrast_acc = self.moco_archive(features_q, features_k, labels, preds_k)
            
            current_epoch = int(self.state.epoch) if self.state.epoch is not None else 0
            if current_epoch >= self.pretrain_epochs:
                current_reward = linear_warmup(
                    current_epoch - self.pretrain_epochs,
                    self.warmup_epochs,
                    self.initial_reward,
                    self.reward
                )
                
                weighted_contrast_loss = (contrast_loss * confidence_sr.detach()).mean()
                
                total_loss = total_loss + weighted_contrast_loss * current_reward
                
                if self.state.global_step % 100 == 0:
                    if hasattr(self, 'log'):
                        self.log({
                            'contrast_loss': contrast_loss.mean().item(),
                            'contrast_acc': contrast_acc,
                            'current_reward': current_reward,
                            'avg_confidence': confidence_sr.mean().item()
                        })
                    else:
                        logger.info(f"Step {self.state.global_step}: contrast_loss={contrast_loss.mean().item():.4f}, "
                                  f"contrast_acc={contrast_acc:.4f}, current_reward={current_reward:.4f}, "
                                  f"avg_confidence={confidence_sr.mean().item():.4f}")
        
        current_epoch = int(self.state.epoch) if self.state.epoch is not None else 0
        if current_epoch >= self.pretrain_epochs and self.model_k is not None:
            self._momentum_update()
        
        if self.moco_archive is not None:
            with torch.no_grad():
                self.moco_archive._dequeue_and_enqueue(features_k, labels, preds_k)
        
        return (total_loss, outputs_q) if return_outputs else total_loss
    
    @torch.no_grad()
    def _momentum_update(self):
        for param_q, param_k in zip(self.model.parameters(), self.model_k.parameters()):
            param_k.data = param_k.data * self.moco_momentum + param_q.data * (1.0 - self.moco_momentum)
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        logger.info(f"Starting epoch {int(state.epoch) if state.epoch is not None else 0}")


def compute_metrics(eval_pred, compute_result=False):
    predictions = eval_pred.predictions if hasattr(eval_pred, "predictions") else eval_pred[0]
    labels = eval_pred.label_ids if hasattr(eval_pred, "label_ids") else eval_pred[1]
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if predictions.shape[1] > 2:
        predictions = predictions[:, :-1]
    preds = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluate_selective_performance(model, eval_dataloader, device, thresholds=[0.7, 0.8, 0.9]):
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in eval_dataloader:
            labels = batch.pop('labels').to(device)
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            outputs = model(**batch)
            logits = outputs['logits']
            
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    if all_logits.shape[1] > 2:
        cls_probs = F.softmax(all_logits[:, :-1], dim=1)
        confidences = torch.max(cls_probs, dim=1)[0]
        predictions = torch.argmax(cls_probs, dim=1)
    else:
        probs = F.softmax(all_logits, dim=1)
        confidences = torch.max(probs, dim=1)[0]
        predictions = torch.argmax(probs, dim=1)
    
    results = {}
    for threshold in thresholds:
        mask = confidences >= threshold
        if mask.sum() > 0:
            selective_preds = predictions[mask]
            selective_labels = all_labels[mask]
            
            accuracy = (selective_preds == selective_labels).float().mean().item()
            coverage = mask.float().mean().item()
            
            results[f'acc@{threshold}'] = accuracy
            results[f'cov@{threshold}'] = coverage
        else:
            results[f'acc@{threshold}'] = 0.0
            results[f'cov@{threshold}'] = 0.0
    return results

def get_model_and_collator(args, tokenizer):
    if args.classifier == "llm2vec":
        model = SelectiveLLM2VecClassifier(args.model_name, num_classes=2, selective=args.use_cclsc)
        collator = LLM2VecCollator(model.encoder)
        use_llm2vec = True
    elif args.classifier == "bert":
        model = BertWithFeatures(model_name="bert-base-uncased", num_labels=2)
        collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')
        use_llm2vec = False
    else:
        raise ValueError(f"Unsupported classifier: {args.classifier}")
    return model, collator, use_llm2vec

def main():
    parser = argparse.ArgumentParser(description='Selective Classification for Unlearning Detection')
    
    parser.add_argument('--model_name', type=str, default='McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp',
                        help='LLM2Vec model name')
    parser.add_argument('--train_data', type=str, required=True, help='Training data path')
    parser.add_argument('--eval_data', type=str, required=True, help='Evaluation data path')
    parser.add_argument('--test_data', type=str, help='Test data path')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--classifier', type=str, default='llm2vec', choices=['llm2vec', 'bert'], help='Classifier type')
    
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')

    parser.add_argument('--use_answer', action='store_true', default=True, 
                        help='Use both question and answer for training (default: True)')
    parser.add_argument('--question_only', dest='use_answer', action='store_false',
                        help='Use only question for training (overrides --use_answer)')
    
    parser.add_argument('--use_weighted_loss', action='store_true',
                        help='Enable weighted cross-entropy loss to handle class imbalance.')
    parser.add_argument('--manual_weights', type=float, nargs=2, default=None,
                    help='Manually specify class weights, e.g., --manual_weights 0.8 1.5')
    parser.add_argument('--use_cclsc', action='store_true', help='Enable CCL-SC algorithm')
    parser.add_argument('--loss_type', type=str, default='csc', 
                        choices=['ce', 'sat', 'csc', 'csc_sat'],
                        help='Loss type')
    parser.add_argument('--moco_k', type=int, default=300, help='MoCo queue size')
    parser.add_argument('--moco_m', type=float, default=0.999, help='MoCo momentum')
    parser.add_argument('--moco_t', type=float, default=0.07, help='MoCo temperature')
    parser.add_argument('--reward', type=float, default=2.2, help='Reward for contrast loss')
    parser.add_argument('--pretrain_epochs', type=int, default=3, help='Pretrain epochs before MoCo')
    parser.add_argument('--warmup_epochs', type=int, default=2, help='Warmup epochs for reward')
    parser.add_argument('--initial_reward', type=float, default=1e-6, help='Initial reward value')

    
    args = parser.parse_args()

    cclsc_flag = "cclsc" if args.use_cclsc else "nocclsc"
    answer_flag = "qa" if args.use_answer else "qonly"
    model_folder = f"{args.model_name.replace('/', '-')}_{cclsc_flag}_{answer_flag}"
    time_str = datetime.datetime.now().strftime("%m%d_%H%M")
    args.output_dir = os.path.join(args.output_dir, model_folder, time_str)

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name if args.classifier == "llm2vec" else "bert-base-uncased",
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if hasattr(tokenizer, "eos_token") else "[PAD]"

    model, data_collator, use_llm2vec = get_model_and_collator(args, tokenizer)
    model.to(device)

    train_dataset = UnlearningDataset(args.train_data, tokenizer, args.max_length, use_llm2vec, args.use_answer)
    eval_dataset = UnlearningDataset(args.eval_data, tokenizer, args.max_length, use_llm2vec, args.use_answer)

    class_weights = None
    if args.manual_weights:
        class_weights = args.manual_weights
    elif args.use_weighted_loss:
        labels = [item['labels'].item() for item in train_dataset]
        num_samples = len(labels)
        num_class_0 = labels.count(0)
        num_class_1 = labels.count(1)

        if num_class_0 == 0 or num_class_1 == 0:
            pass
        else:
            weight_for_0 = num_samples / (2.0 * num_class_0)
            weight_for_1 = num_samples / (2.0 * num_class_1)
            class_weights = [weight_for_0, weight_for_1]

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_dir=f'{args.output_dir}/logs',
        logging_steps=100,
        eval_steps=1000,
        save_steps=1000,
        evaluation_strategy='steps',
        save_strategy='steps',
        greater_is_better=True,
        warmup_steps=50,
        bf16=torch.cuda.is_available(),
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model='eval_f1',
        max_grad_norm=1.0,
    )
    
    if args.use_cclsc:
        model_k = copy.deepcopy(model)
        model_k.to(device)
        if args.classifier == "bert":
            feature_dim = model.bert.config.hidden_size
        else:
            feature_dim = model.encoder.model.config.hidden_size
        moco_archive = TextMoCo(
            dim=feature_dim,
            K=args.moco_k,
            m=args.moco_m,
            T=args.moco_t
        )
        moco_archive.to(device)
        trainer = SelectiveTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            class_weights=class_weights,
            moco_archive=moco_archive,
            model_k=model_k,
            loss_type=args.loss_type,
            reward=args.reward,
            pretrain_epochs=args.pretrain_epochs,
            warmup_epochs=args.warmup_epochs,
            initial_reward=args.initial_reward,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
    else:
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            class_weights=class_weights,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

    
    trainer.train()
    
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    log_history = trainer.state.log_history

    train_logs = [log for log in log_history if 'loss' in log and 'epoch' in log]
    with open(os.path.join(args.output_dir, 'train_logs.json'), 'w') as f:
        json.dump(train_logs, f, indent=2)

    contrast_logs = [log for log in log_history if 'contrast_loss' in log]
    if contrast_logs:
        with open(os.path.join(args.output_dir, 'contrast_logs.json'), 'w') as f:
            json.dump(contrast_logs, f, indent=2)

    eval_logs = [log for log in log_history if 'eval_loss' in log]
    with open(os.path.join(args.output_dir, 'eval_logs.json'), 'w') as f:
        json.dump(eval_logs, f, indent=2)

    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=args.batch_size, 
        collate_fn=data_collator
    )
    
    selective_results = evaluate_selective_performance(
        model, eval_dataloader, device,
        thresholds=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1]
    )
    
    with open(f'{args.output_dir}/selective_results.json', 'w') as f:
        json.dump(selective_results, f, indent=2)
    
    if args.test_data:
        test_dataset = UnlearningDataset(args.test_data, tokenizer, args.max_length, use_llm2vec, args.use_answer)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            collate_fn=data_collator
        )
        all_logits = []
        all_labels = []
        model.eval()
        with torch.no_grad():
            for batch in test_dataloader:
                labels = batch.pop('labels').to(device)
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs['logits']
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
        all_logits = torch.cat(all_logits, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        preds = np.argmax(all_logits, axis=1)
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
        accuracy = accuracy_score(all_labels, preds)
        f1 = f1_score(all_labels, preds, average='binary')
        precision, recall, f1_detail, support = precision_recall_fscore_support(
            all_labels, preds, average=None, labels=[0, 1]
        )
        test_results = {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1_detail': f1_detail.tolist(),
            'support': support.tolist()
        }
        with open(f'{args.output_dir}/test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
            
if __name__ == "__main__":
    main()
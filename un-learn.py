import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
import numpy as np
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from transformers import BitsAndBytesConfig
from typing import List, Optional, Union

class LlamaSemiSupervised:
    def __init__(
        self, 
        model_name: str = "meta-llama/Llama-2-7b-hf",  # Change this to new version when available
        num_labels: int = 2,
        batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 2e-4,
        max_length: int = 512,
        lora_r: int = 16,
        lora_alpha: int = 32,
        device: Optional[str] = None
    ):
        """
        Initialize the semi-supervised learning pipeline for Llama models.
        
        Args:
            model_name: Name or path of the Llama model
            num_labels: Number of classification labels
            batch_size: Training batch size
            num_epochs: Number of training epochs
            learning_rate: Learning rate for training
            max_length: Maximum sequence length
            lora_r: Rank of LoRA adaptations
            lora_alpha: Alpha parameter for LoRA
            device: Device to use for training ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize quantization config
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False
        )
        
        # Initialize tokenizer with version-agnostic approach
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Handle tokenizer pad token
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Initialize model with quantization
        self.model = self._initialize_model()
        
        # Configure LoRA
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=self._get_target_modules(),
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_CLS
        )
        
        # Prepare model with LoRA
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()
        
    def _get_target_modules(self) -> List[str]:
        """
        Get target modules for LoRA based on model architecture.
        Adapts to different Llama versions.
        """
        # Default Llama-2 modules
        base_modules = ["q_proj", "v_proj"]
        
        # Try to detect model architecture and adjust accordingly
        try:
            model_modules = [name for name, _ in self.model.named_modules()]
            if any("self_attn" in module for module in model_modules):
                return ["self_attn.q_proj", "self_attn.v_proj"]
            return base_modules
        except Exception:
            return base_modules

    def _initialize_model(self):
        """Initialize the model with proper configuration"""
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            quantization_config=self.bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        return prepare_model_for_kbit_training(model)
    
    def prepare_data(self, texts: List[str], labels: Optional[List[int]] = None) -> Dataset:
        """Prepare data for training or inference with proper handling of long sequences"""
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        if labels is not None:
            return Dataset.from_dict({
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'labels': labels
            })
        return Dataset.from_dict({
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask']
        })
    
    def train(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None
    ):
        """Train the model with optimized settings"""
        train_dataset = self.prepare_data(train_texts, train_labels)
        
        if val_texts is not None and val_labels is not None:
            val_dataset = self.prepare_data(val_texts, val_labels)
        else:
            train_dataset, val_dataset = train_test_split(
                train_dataset,
                test_size=0.1,
                random_state=42
            )
        
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=4,
            warmup_ratio=0.1,
            learning_rate=self.learning_rate,
            fp16=True,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            optim="paged_adamw_32bit",
            gradient_checkpointing=True,
            # Add auto device mapping
            device=self.device
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorWithPadding(self.tokenizer)
        )
        
        trainer.train()
    
    def predict(self, texts: List[str]) -> torch.Tensor:
        """Generate predictions with proper error handling"""
        dataset = self.prepare_data(texts)
        
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer
        )
        
        try:
            with torch.no_grad():
                predictions = trainer.predict(dataset)
            return torch.nn.functional.softmax(
                torch.tensor(predictions.predictions),
                dim=1
            )
        except Exception as e:
            print(f"Prediction error: {e}")
            raise
    
    def semi_supervised_learning(
        self,
        labelled_texts: List[str],
        labelled_labels: List[int],
        unlabelled_texts: List[str],
        confidence_threshold: float = 0.9
    ) -> torch.Tensor:
        """Run the complete semi-supervised learning pipeline"""
        print(f"Initial training on {len(labelled_texts)} labelled examples...")
        self.train(labelled_texts, labelled_labels)
        
        print("Generating predictions for unlabelled data...")
        probabilities = self.predict(unlabelled_texts)
        predictions = torch.argmax(probabilities, dim=1)
        max_probs = torch.max(probabilities, dim=1)[0]
        
        confident_idx = max_probs >= confidence_threshold
        new_labels = predictions[confident_idx]
        new_texts = [unlabelled_texts[i] for i in range(len(unlabelled_texts)) 
                    if confident_idx[i]]
        
        print(f"Found {len(new_texts)} high-confidence predictions...")
        
        all_texts = labelled_texts + new_texts
        all_labels = labelled_labels + new_labels.tolist()
        
        print(f"Retraining on combined dataset of {len(all_texts)} examples...")
        self.train(all_texts, all_labels)
        
        final_probabilities = self.predict(unlabelled_texts)
        return torch.argmax(final_probabilities, dim=1)

def main():
    """Example usage of the pipeline"""
    # Sample data
    labelled_texts = [
        "This product exceeded my expectations!",
        "Terrible customer service, would not recommend."
    ]
    labelled_labels = [1, 0]
    unlabelled_texts = [
        "Amazing quality and fast shipping!",
        "The product broke after one use.",
        "Pretty good value for money."
    ]
    
    # Initialize model
    ssl_model = LlamaSemiSupervised(
        model_name="meta-llama/Llama-2-7b-hf",  # Update this when new version is available
        num_labels=2,
        batch_size=4
    )
    
    # Run semi-supervised learning
    predictions = ssl_model.semi_supervised_learning(
        labelled_texts,
        labelled_labels,
        unlabelled_texts
    )
    
    print("Final predictions:", predictions)

if __name__ == "__main__":
    main()
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import evaluate
import torch
import numpy as np

df = pd.read_csv("your_dataset.csv") 

# Format input prompt for causal LM
df["prompt"] = df.apply(lambda row: f"Act as Indian legal expert to Summarize: {row['judgment']}\nSummary:", axis=1)
df["target"] = df["head note"]

# Convert to Hugging Face dataset
dataset = Dataset.from_pandas(df[["prompt", "target"]])
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Load model and tokenizer
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # For causal LM compatibility
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenization
def preprocess(example):
    input_ids = tokenizer(
        example["prompt"], padding="max_length", truncation=True, max_length=512
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["target"], padding="max_length", truncation=True, max_length=128
        )
    input_ids["labels"] = labels["input_ids"]
    return input_ids

tokenized_train = train_dataset.map(preprocess, batched=True)
tokenized_eval = eval_dataset.map(preprocess, batched=True)

# ROUGE metric
rouge = evaluate.load("rouge")

def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 2) for k, v in result.items()}
    return result

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    logging_dir="./logs",
    logging_steps=20,
)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()
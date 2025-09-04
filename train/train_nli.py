from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd

# 1. Load your data
df = pd.read_csv('D:/Project/Hackathon/dataset/nli_pairs.csv')  # columns: premise, hypothesis, label

# Map your binary labels to 3-class NLI labels
# 0 (fake) -> 2 (contradiction), 1 (real) -> 0 (entailment)
df['label'] = df['label'].map({1: 0, 0: 2})

# 2. Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_ds = dataset['train']
eval_ds = dataset['test']

# 3. Tokenizer and model
model_name = 'roberta-large-mnli'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def preprocess(examples):
    return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding='max_length', max_length=128)

train_ds = train_ds.map(preprocess, batched=True)
eval_ds = eval_ds.map(preprocess, batched=True)
train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
eval_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# 4. Training arguments
training_args = TrainingArguments(
    output_dir='D:/Project/Hackathon/models/nli_roberta',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='D:/Project/Hackathon/logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss'
)

# 5. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
)

# 6. Train
trainer.train()

# 7. Save the model
model.save_pretrained('D:/Project/Hackathon/models/nli_roberta')
tokenizer.save_pretrained('D:/Project/Hackathon/models/nli_roberta')
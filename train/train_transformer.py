import sys
import transformers
print('Transformers module path:', transformers.__file__)
print('Transformers version:', transformers.__version__)
print('sys.path:', sys.path)

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset
import torch

# Set absolute paths
TRUE_PATH = 'D:/Project/Hackathon/dataset/True.csv'
FAKE_PATH = 'D:/Project/Hackathon/dataset/Fake.csv'
MODEL_SAVE_PATH = 'D:/Project/Hackathon/models/bert_fakenews'
LOGGING_PATH = 'D:/Project/Hackathon/logs'

# 1. Load and label the ISOT dataset
true_df = pd.read_csv(TRUE_PATH)
fake_df = pd.read_csv(FAKE_PATH)
true_df['label'] = 1
fake_df['label'] = 0
df = pd.concat([true_df, fake_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Use both title and text if available, else just text
if 'title' in df.columns and 'text' in df.columns:
    df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
elif 'text' in df.columns:
    df['content'] = df['text'].fillna('')
elif 'content' in df.columns:
    df['content'] = df['content'].fillna('')
else:
    raise ValueError("No suitable text column found in the dataset.")

# 2. Split into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# 3. Convert to HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df[['content', 'label']])
val_dataset = Dataset.from_pandas(val_df[['content', 'label']])

# 4. Tokenizer and model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    return tokenizer(examples['content'], truncation=True, padding='max_length', max_length=256)

train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

# 5. Set format for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# 6. Model
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 7. Training arguments
training_args = TrainingArguments(
    output_dir=MODEL_SAVE_PATH,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir=LOGGING_PATH,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    fp16=torch.cuda.is_available(),  # Use mixed precision if GPU is available
    dataloader_pin_memory=torch.cuda.is_available(),
    report_to="none"
)

# 8. Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 9. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# 10. Train
trainer.train()

# 11. Save the model
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH) 
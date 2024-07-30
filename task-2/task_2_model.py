import pandas as pd
import jsonlines
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, get_scheduler
from torch.utils.data import DataLoader, Dataset
from datasets import Dataset as HFDataset
from tqdm.auto import tqdm
import torch

# Function to load JSON Lines files
def load_jsonl(file_path):
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    return pd.DataFrame(data)

# Load the datasets
train_df = load_jsonl('data/train.jsonl')
val_df = load_jsonl('data/val.jsonl')
test_df = load_jsonl('data/test.jsonl')

# Combine postText and targetParagraphs into input for the model
train_df['input_text'] = train_df.apply(lambda row: ' '.join(row['postText']) + ' ' + ' '.join(row['targetParagraphs']), axis=1)
val_df['input_text'] = val_df.apply(lambda row: ' '.join(row['postText']) + ' ' + ' '.join(row['targetParagraphs']), axis=1)
test_df['input_text'] = test_df.apply(lambda row: ' '.join(row['postText']) + ' ' + ' '.join(row['targetParagraphs']), axis=1)

# Define the Spoiler Dataset
class SpoilerDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

# Tokenize the inputs and targets
tokenizer = T5Tokenizer.from_pretrained('t5-small')

def tokenize_function(examples):
    model_inputs = tokenizer(examples['input_text'], max_length=512, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['spoiler'], max_length=50, truncation=True, padding="max_length")
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# Convert to Hugging Face Datasets
hf_train_dataset = HFDataset.from_pandas(train_df[['input_text', 'spoiler']])
hf_val_dataset = HFDataset.from_pandas(val_df[['input_text', 'spoiler']])

# Tokenize datasets
tokenized_train_dataset = hf_train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = hf_val_dataset.map(tokenize_function, batched=True)

# Convert to PyTorch Datasets
train_dataset = SpoilerDataset(tokenized_train_dataset)
val_dataset = SpoilerDataset(tokenized_val_dataset)

# Load pre-trained T5 model for sequence generation
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Prepare the data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Set up the optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 1
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Move the model to the GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Training loop
progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

# Validation loop
model.eval()
val_predictions = []
for batch in val_loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], max_length=50)
    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    val_predictions.extend(predictions)

val_df['predicted_spoiler'] = val_predictions

# Prepare the test dataset
tokenized_test_dataset = tokenizer(test_df['input_text'].tolist(), max_length=512, truncation=True, padding="max_length", return_tensors="pt")
test_dataset = SpoilerDataset(tokenized_test_dataset)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Make predictions on the test set
model.eval()
test_predictions = []
for batch in test_loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], max_length=50)
    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    test_predictions.extend(predictions)

# Prepare the submission file
submission = pd.DataFrame({
    'id': range(len(test_predictions)),  # Use a range of integers as the IDs
    'spoiler': test_predictions
})

# Save the predictions to a CSV file
submission.to_csv('submission.csv', index=False)

# AL_Bald_Sampling.py

import os
import sys
from datetime import datetime
import torch
import transformers
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from statistics import mean
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import csv
import argparse

# Increase CSV field size limit
maxInt = sys.maxsize
csv.field_size_limit(maxInt)

# Initialize the parser
parser = argparse.ArgumentParser(description="Run Active Learning with BALD Sampling")

# Add arguments
parser.add_argument('--output', type=str, required=True, help="Output file name for logging results")
parser.add_argument('--folder', type=str, required=True, help="Folder containing the train.tsv and test.tsv files")
parser.add_argument('--pretrain', type=str, required=True, help="Pre-trained model path")

# Parse arguments
args = parser.parse_args()

# Extract the folder, output, and pretrain model paths
folder = args.folder
output_file = args.output
pretrain_model = args.pretrain

# Enable CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to apply attention mask
def apply_attention_mask(x):
    return x.ne(tokenizer.pad_token_id).to(int)

# Define the folder and log file path
log_folder = os.path.join(folder, 'Logs')  # Folder for logs
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
    print(f"Logs folder created at: {log_folder}")

# Generate a timestamp string
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Define the result file name with the current timestamp
log_file_path = os.path.join(log_folder, f'{output_file}_{timestamp}.txt')
print(f"Log file will be saved to: {log_file_path}")

# Load training data from the provided folder
train_file = os.path.join(folder, 'train.tsv')
train_data = []
with open(train_file, 'r', encoding='utf-8') as fd:
    rd = csv.reader(fd, delimiter="\t", quotechar='"')
    next(rd)  # Skip the header
    for row in rd:
        text = row[0]
        labels = int(row[1])
        train_data.append([text, labels])
train_df = pd.DataFrame(train_data, columns=['text', 'label'])

# Load test data from the provided folder
test_file = os.path.join(folder, 'test.tsv')
test_data = []
with open(test_file, 'r', encoding='utf-8') as fd:
    rd = csv.reader(fd, delimiter="\t", quotechar='"')
    next(rd)  # Skip the header
    for row in rd:
        text = row[0]
        labels = int(row[1])
        test_data.append([text, labels])
test_df = pd.DataFrame(test_data, columns=['text', 'label'])

# Define the model using the pre-trained model path provided
model = transformers.BertForSequenceClassification.from_pretrained(
    pretrain_model, num_labels=2, output_attentions=False, output_hidden_states=False
)
tokenizer = transformers.BertTokenizer.from_pretrained(pretrain_model)

# Move the model to the device (e.g., GPU if available)
model.to(device)

# Tokenize function
def tokenize_texts(texts):
    return tokenizer(
        texts,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=512
    )

# Tokenize test data
print("Tokenizing test data...")
tokenized_test = tokenize_texts(test_df['text'].tolist())
x_test = tokenized_test['input_ids']
test_attention_mask = tokenized_test['attention_mask']
y_test = torch.tensor(test_df['label'].values)

# Create test DataLoader
test_dataset = TensorDataset(x_test, test_attention_mask, y_test)
test_loader = DataLoader(
    test_dataset, batch_size=10, shuffle=False, num_workers=4, pin_memory=True
)

# Function to run the test
def test(device, model, test_loader, annotation_round):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for x_batch, attention_mask_batch, y_batch in test_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            attention_mask_batch = attention_mask_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            out = model(x_batch, attention_mask=attention_mask_batch)
            preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
            labels.extend(y_batch.cpu().numpy())
    model.train()

    predicted = np.array(preds)
    actual = np.array(labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, average='weighted', zero_division=0
    )

    # Log results to file
    result_file = f'{args.output}.txt'
    try:
        with open("Logs/"+result_file, 'a') as f:
            f.write(f"Annotation Round {annotation_round}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1: {f1:.4f}\n")
            f.write(f"-------------------------------------\n")
        print(f"Results saved to: {log_file_path}")
    except Exception as e:
        print(f"Error writing log file: {e}")

# Split the data into labeled and unlabeled sets
x_labeled_df, x_unlabeled_df = train_test_split(
    train_df, test_size=0.99, stratify=train_df['label'], random_state=42
)

# Tokenize labeled data
print("Tokenizing labeled data...")
tokenized_labeled = tokenize_texts(x_labeled_df['text'].tolist())
x_train_labeled = tokenized_labeled['input_ids']
attention_mask_labeled = tokenized_labeled['attention_mask']
y_train_labeled = torch.tensor(x_labeled_df['label'].values)

# Create labeled DataLoader
batch_size = 10  # Updated batch_size to 10
unlabeled_batch_size = 64  # Increased batch size for unlabeled data
labeled_dataset = TensorDataset(x_train_labeled, attention_mask_labeled, y_train_labeled)
labeled_loader = DataLoader(
    labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
)

# Tokenize unlabeled data
print("Tokenizing unlabeled data...")
tokenized_unlabeled = tokenize_texts(x_unlabeled_df['text'].tolist())
x_train_unlabeled = tokenized_unlabeled['input_ids']
attention_mask_unlabeled = tokenized_unlabeled['attention_mask']

# Create indices for the unlabeled data
unlabeled_indices = np.arange(len(x_train_unlabeled))

# Create unlabeled DataLoader with indices
unlabeled_dataset = TensorDataset(x_train_unlabeled, attention_mask_unlabeled, torch.tensor(unlabeled_indices))
unlabeled_loader = DataLoader(
    unlabeled_dataset, batch_size=unlabeled_batch_size, shuffle=False, num_workers=4, pin_memory=True
)

# Active Learning loop
optim = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
num_samples_to_select = 10
num_annotation_rounds = 110  # Updated from 200 to 110
num_epochs = 10
num_inference_cycles = 5  # Number of stochastic forward passes for BALD

# Initialize AMP scaler
scaler = GradScaler()

for annotation_round in range(num_annotation_rounds):
    print(f"Annotation Round {annotation_round + 1}")

    # Train the model on the currently labeled set of data
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        for x_batch, attention_mask_batch, y_batch in tqdm(
            labeled_loader, desc=f"Epoch {epoch + 1}"
        ):
            x_batch = x_batch.to(device, non_blocking=True)
            attention_mask_batch = attention_mask_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optim.zero_grad()
            with autocast():
                out = model(
                    x_batch,
                    attention_mask=attention_mask_batch,
                    labels=y_batch
                )
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            epoch_losses.append(loss.item())
        print(f"Epoch {epoch + 1} loss: {mean(epoch_losses):.4f}")

    # BALD sampling
    # Use the trained model to make predictions on the unlabeled set
    # Perform multiple stochastic forward passes to estimate mutual information

    mutual_info_scores = []
    all_indices = []

    model.train()  # Enable dropout during inference

    with torch.no_grad():
        for x_batch_unlabeled, attention_mask_unlabeled_batch, batch_indices in tqdm(
            unlabeled_loader, desc="Unlabeled Data"
        ):
            x_batch_unlabeled = x_batch_unlabeled.to(device, non_blocking=True)
            attention_mask_unlabeled_batch = attention_mask_unlabeled_batch.to(device, non_blocking=True)
            batch_indices = batch_indices.numpy()
    
            T = num_inference_cycles
    
            # Vectorized forward passes
            x_batch_expanded = x_batch_unlabeled.unsqueeze(0).expand(T, -1, -1)
            attention_mask_expanded = attention_mask_unlabeled_batch.unsqueeze(0).expand(T, -1, -1)
    
            x_batch_flat = x_batch_expanded.reshape(-1, x_batch_unlabeled.size(1))
            attention_mask_flat = attention_mask_expanded.reshape(-1, attention_mask_unlabeled_batch.size(1))
    
            out = model(
                x_batch_flat,
                attention_mask=attention_mask_flat
            )
            logits = out.logits
            probs = F.softmax(logits, dim=1)
            probs = probs.view(T, -1, probs.size(-1))
    
            # Compute mutual information as before
            mean_probs = probs.mean(dim=0)
            predictive_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)
            expected_entropy = -torch.mean(torch.sum(probs * torch.log(probs + 1e-10), dim=2), dim=0)
            mutual_info = predictive_entropy - expected_entropy
    
            mutual_info_scores.extend(mutual_info.cpu().numpy())
            all_indices.extend(batch_indices)

    mutual_info_scores = np.array(mutual_info_scores)
    all_indices = np.array(all_indices)

    # Get indices of samples with highest mutual information
    sorted_indices = all_indices[np.argsort(mutual_info_scores)[::-1]]

    selected_indices = sorted_indices[:num_samples_to_select]

    print(f"Adding {num_samples_to_select} labels")

    # Update labeled and unlabeled datasets
    new_labeled_data = x_unlabeled_df.iloc[selected_indices]
    x_labeled_df = pd.concat([x_labeled_df, new_labeled_data], ignore_index=True)
    x_unlabeled_df = x_unlabeled_df.drop(selected_indices).reset_index(drop=True)

    # Tokenize new labeled data
    tokenized_new_labeled = tokenize_texts(new_labeled_data['text'].tolist())
    x_new_labeled = tokenized_new_labeled['input_ids']
    attention_mask_new_labeled = tokenized_new_labeled['attention_mask']
    y_new_labeled = torch.tensor(new_labeled_data['label'].values)

    # Update labeled dataset and DataLoader
    x_train_labeled = torch.cat([x_train_labeled, x_new_labeled], dim=0)
    attention_mask_labeled = torch.cat([attention_mask_labeled, attention_mask_new_labeled], dim=0)
    y_train_labeled = torch.cat([y_train_labeled, y_new_labeled], dim=0)

    labeled_dataset = TensorDataset(x_train_labeled, attention_mask_labeled, y_train_labeled)
    labeled_loader = DataLoader(
        labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    # Remove selected samples from the unlabeled dataset
    mask = np.ones(len(x_train_unlabeled), dtype=bool)
    mask[selected_indices] = False

    x_train_unlabeled = x_train_unlabeled[mask]
    attention_mask_unlabeled = attention_mask_unlabeled[mask]
    x_unlabeled_df = x_unlabeled_df.reset_index(drop=True)

    # Update indices for the unlabeled data
    unlabeled_indices = np.arange(len(x_train_unlabeled))

    # Recreate the unlabeled dataset and DataLoader
    unlabeled_dataset = TensorDataset(x_train_unlabeled, attention_mask_unlabeled, torch.tensor(unlabeled_indices))
    unlabeled_loader = DataLoader(
        unlabeled_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    if (annotation_round + 1) % 5 == 0:
        test(device, model, test_loader, annotation_round + 1)

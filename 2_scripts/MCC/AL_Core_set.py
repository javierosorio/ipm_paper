import os
import sys
from datetime import datetime
import torch
import transformers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder as OHE
from torch.optim import AdamW
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from statistics import mean
import torch.nn.functional as F
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
import csv
import argparse

# Initialize the parser
parser = argparse.ArgumentParser(description="Run Active Learning Core Set Script")

# Add arguments
parser.add_argument('--output', type=str, required=True, help="Output File name for logging results")
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
def apply_attention_mask(x, tokenizer):
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
    pretrain_model, num_labels=5, output_attentions=False, output_hidden_states=True  # output_hidden_states=True is crucial for CLS embeddings
)
tokenizer = transformers.BertTokenizer.from_pretrained(pretrain_model, max_length=512)

# Move the model to the device (e.g., GPU if available)
model.to(device)
model.train()

# Preprocessing
encoder = OHE(sparse=False)
encoder.fit(train_df[['label']])

# Tokenize function
def tokenize(samples: 'list[str]'):
    return tokenizer(samples, return_tensors='pt', padding='max_length', truncation=True, max_length=512).input_ids

# Tokenize test data and move to the device
x_test = tokenize(test_df['text'].tolist()).to(device)

# Function to run the test
def test(device, apply_attention_mask, test_df, model, tokenizer, batch_size, annotation_round):
    preds = []
    with torch.no_grad():
        for i in range(0, x_test.shape[0], batch_size):
            x = x_test[i : i + batch_size].to(device)
            out = model(x, attention_mask=apply_attention_mask(x, tokenizer))
            preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())

    predicted = list(preds)
    actual = test_df['label'].tolist()
    precision, recall, f1, _ = precision_recall_fscore_support(actual, predicted, average='weighted')

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
x_labeled, x_unlabeled = train_test_split(train_df, test_size=0.99, stratify=train_df['label'])
x_train_labeled = tokenize(x_labeled['text'].tolist()).to(device)
x_train_unlabeled = tokenize(x_unlabeled['text'].tolist()).to(device)

# Ensure the label data (target) is moved to the same device (GPU)
y_train_labeled = torch.Tensor(encoder.transform(x_labeled[['label']])).to(device)
y_train_unlabeled = torch.Tensor(encoder.transform(x_unlabeled[['label']])).to(device)

# Active Learning loop
optim = AdamW(model.parameters(), lr=1e-5, eps=1e-8)
num_samples_to_select = 10
num_annotation_rounds = 150
num_epochs = 10
batch_size = 32

for annotation_round in range(num_annotation_rounds):
    print(f"Annotation Round {annotation_round + 1}")

    # Train the model on the currently labeled set of data
    for epoch in range(num_epochs):
        epoch_losses = []
        for i in tqdm(range(0, x_train_labeled.shape[0], batch_size), desc=f"Epoch {epoch + 1}"):
            x = x_train_labeled[i : i + batch_size]
            y = y_train_labeled[i : i + batch_size]
            model.zero_grad()
            out = model(x, attention_mask=apply_attention_mask(x, tokenizer), labels=y)
            epoch_losses.append(out.loss.item())
            out.loss.backward()
            optim.step()
        print(f"Epoch {epoch + 1} loss: {mean(epoch_losses)}")

    # Extract CLS token embeddings for both labeled and unlabeled data
    print("Extracting CLS token embeddings for labeled and unlabeled data...")
    cls_vector_labeled = []
    cls_vector_unlabeled = []

    # Labeled data embeddings
    with torch.no_grad():
        for i in range(0, x_train_labeled.shape[0], batch_size):
            x = x_train_labeled[i: i + batch_size]
            outputs = model(x, attention_mask=apply_attention_mask(x, tokenizer))
            hidden_states = outputs.hidden_states[-1]  # CLS token is at index 0
            cls_vector_labeled.extend(hidden_states[:, 0, :].cpu().numpy())

    # Unlabeled data embeddings
    with torch.no_grad():
        for i in range(0, x_train_unlabeled.shape[0], batch_size):
            x = x_train_unlabeled[i: i + batch_size]
            outputs = model(x, attention_mask=apply_attention_mask(x, tokenizer))
            hidden_states = outputs.hidden_states[-1]  # CLS token is at index 0
            cls_vector_unlabeled.extend(hidden_states[:, 0, :].cpu().numpy())

    cls_vector_labeled = torch.Tensor(cls_vector_labeled)
    cls_vector_unlabeled = torch.Tensor(cls_vector_unlabeled)

    # Core-Set strategy: Compute pairwise distances between labeled and unlabeled data
    print("Computing distances between labeled and unlabeled embeddings...")
    selected_indices = []
    for _ in range(num_samples_to_select):
        distances = pairwise_distances_argmin_min(cls_vector_unlabeled, cls_vector_labeled)[1]
        max_distance_index = torch.argmax(torch.Tensor(distances))
        selected_indices.append(max_distance_index.item())
        cls_vector_labeled = torch.cat([cls_vector_labeled, cls_vector_unlabeled[max_distance_index].unsqueeze(0)])
        cls_vector_unlabeled = torch.cat([cls_vector_unlabeled[:max_distance_index], cls_vector_unlabeled[max_distance_index+1:]])

    print(f"Selected indices: {selected_indices}")

    # Add the selected indices to the labeled set
    print(f"Adding {num_samples_to_select} labels")
    x_train_labeled = torch.cat([x_train_labeled, x_train_unlabeled[selected_indices].to(device)])
    y_train_labeled = torch.cat([y_train_labeled, y_train_unlabeled[selected_indices].to(device)])

    # Efficiently filter out least confident samples
    mask = torch.ones(x_train_unlabeled.shape[0], dtype=torch.bool, device=device)
    mask[selected_indices] = False
    x_train_unlabeled = x_train_unlabeled[mask]
    y_train_unlabeled = y_train_unlabeled[mask]

    if (annotation_round+1) % 5 == 0 :
        test(device, apply_attention_mask, test_df, model, tokenizer, batch_size, annotation_round+1)

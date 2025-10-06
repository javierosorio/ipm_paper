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
import csv
import argparse

csv.field_size_limit(160000)  # You can set a larger limit if needed
# Initialize the parser to handle command-line arguments
parser = argparse.ArgumentParser(description="Run Margin Sampling for Active Learning")
parser.add_argument('--output', type=str, required=True, help="Output file name for logging results")
parser.add_argument('--folder', type=str, required=True, help="Input folder containing the train1.tsv and test1.tsv files")
parser.add_argument('--pretrain', type=str, required=True, help="Path to the pre-trained model")

# Parse the arguments
args = parser.parse_args()

# Use parsed arguments in the script
output_file = args.output
folder = args.folder
pretrain_model = args.pretrain

# Enable CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to apply attention mask
def apply_attention_mask(x, tokenizer):
    return x.ne(tokenizer.pad_token_id).to(int)

# Define the log folder and ensure it exists
log_folder = os.path.join(folder, 'Logs')
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

# Generate a timestamp string
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Define the result file name with the current timestamp
result_file = f'{args.output}_{timestamp}.txt'

# Full path for the log file in the Logs folder
log_file_path = os.path.join(log_folder, result_file)

# Load training data
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

# Load test data
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

# Define your model
model = transformers.BertForSequenceClassification.from_pretrained(
    pretrain_model, num_labels=2, output_attentions=False, output_hidden_states=False
)
tokenizer = transformers.BertTokenizer.from_pretrained(pretrain_model, max_length=512)
model.train()
model.to(device)

# Preprocessing
encoder = OHE(sparse=False)
encoder.fit(train_df[['label']])

# Tokenize function
def tokenize(samples: 'list[str]'):
    return tokenizer(samples, return_tensors='pt', padding='max_length', truncation=True, max_length=512).input_ids

# Tokenize test data
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
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(actual, predicted, average='weighted')
    precision_mi, recall_mi, f1_mi, _ = precision_recall_fscore_support(actual, predicted, average='micro')
    precision_ma, recall_ma, f1_ma, _ = precision_recall_fscore_support(actual, predicted, average='macro')

    print(f"Precision weighted: {precision_w:.4f}")
    print(f"Recall weighted: {recall_w:.4f}")
    print(f"F1 weighted: {f1_w:.4f}")

    print(f"Precision micro: {precision_mi:.4f}")
    print(f"Recall micro: {recall_mi:.4f}")
    print(f"F1 micro: {f1_mi:.4f}")

    print(f"Precision macro: {precision_ma:.4f}")
    print(f"Recall macro: {recall_ma:.4f}")
    print(f"F1 macro: {f1_ma:.4f}")


    # Log results to file in the Logs folder
    try:
        result_file = f'{args.output}.txt'
        with open("Logs/"+result_file, 'a') as f:
            sys.stdout = f  # Redirect print output to the file
            print(f"Annotation Round {annotation_round}")

            print(f"Precision weighted: {precision_w:.4f}")
            print(f"Recall weighted: {recall_w:.4f}")
            print(f"F1: {f1_w:.4f}")

            # print(f"Precision micro: {precision_mi:.4f}")
            # print(f"Recall micro: {recall_mi:.4f}")
            # print(f"F1 micro: {f1_mi:.4f}")

            # print(f"Precision macro: {precision_ma:.4f}")
            # print(f"Recall macro: {recall_ma:.4f}")
            # print(f"F1 macro: {f1_ma:.4f}")
        sys.stdout = sys.__stdout__  # Reset to console output
        print(f"Results saved to: {log_file_path}")
    except Exception as e:
        print(f"Error writing log file: {e}")

# Split the data into labeled and unlabeled sets
x_labeled, x_unlabeled = train_test_split(train_df, test_size=.99, stratify=train_df['label'])
x_train_labeled = tokenize(x_labeled['text'].tolist()).to(device)
x_train_unlabeled = tokenize(x_unlabeled['text'].tolist()).to(device)

y_train_labeled = torch.Tensor(encoder.transform(x_labeled[['label']])).to(device)
y_train_unlabeled = torch.Tensor(encoder.transform(x_unlabeled[['label']])).to(device)

# Active Learning loop
optim = AdamW(model.parameters(), lr=1e-5, eps=1e-8)
num_samples_to_select = 10
num_annotation_rounds = 110
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

    # Use the trained model to make predictions on the unlabeled set
    margin_scores = []
    with torch.no_grad():
        model.eval()
        for i in range(0, x_train_unlabeled.shape[0], batch_size):
            x_batch_unlabeled = x_train_unlabeled[i : i + batch_size].to(device)
            out = model(x_batch_unlabeled, attention_mask=apply_attention_mask(x_batch_unlabeled, tokenizer))
            logits = out.logits
            probs = F.softmax(logits, dim=1)
            
            # Calculate margin (difference between the top two probabilities)
            top_two_probs, _ = torch.topk(probs, 2, dim=1)
            margins = top_two_probs[:, 0] - top_two_probs[:, 1]
            margin_scores.extend(margins.cpu().numpy())

    # Select samples with the smallest margin (most uncertain)
    sorted_indices = sorted(range(len(margin_scores)), key=lambda i: margin_scores[i])
    selected_indices = sorted_indices[:num_samples_to_select]
    print(f"Selected indices: {selected_indices}")
    print(f"Selected margins: {[margin_scores[i] for i in selected_indices]}")

    # Add the selected indices to the labeled set
    print(f"Adding {num_samples_to_select} labels")
    
    # Update labeled and unlabeled sets
    x_train_labeled = torch.cat([x_train_labeled, x_train_unlabeled[selected_indices].to(device)])
    y_train_labeled = torch.cat([y_train_labeled, y_train_unlabeled[selected_indices].to(device)])
    
    # Efficiently filter out least confident samples
    mask = torch.ones(x_train_unlabeled.shape[0], dtype=torch.bool, device=device)
    mask[selected_indices] = False
    x_train_unlabeled = x_train_unlabeled[mask]
    y_train_unlabeled = y_train_unlabeled[mask]

    # Test model every round
    if (annotation_round+1) % 5 == 0 :
        test(device, apply_attention_mask, test_df, model, tokenizer, batch_size, annotation_round + 1)

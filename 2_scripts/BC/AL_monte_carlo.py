import pandas as pd
import json
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OneHotEncoder as OHE
import transformers
import torch
from torch.optim import AdamW
from tqdm import tqdm
from statistics import mean
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
import csv
import sys
from datetime import datetime
import argparse
import torch.nn as nn

csv.field_size_limit(160000)  # You can set a larger limit if needed
# Initialize the parser
parser = argparse.ArgumentParser(description="Process some input.")

# Add arguments
parser.add_argument('--output', type=str, required=True, help="Output File name")
parser.add_argument('--folder', type=str, required=True, help="Input Folder name")
parser.add_argument('--pretrain', type=str, required=True, help="Input pretrain model")

# Parse arguments
args = parser.parse_args()
# Use the parsed arguments
print(f"Output file, {args.output} will be generated.")

# Enable CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def apply_attention_mask(x, tokenizer):
    return x.ne(tokenizer.pad_token_id).to(int)

train_data = []

with open(f'{args.folder}'+'/train.tsv') as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            # Skip the header
            next(rd) 
            for row in rd:
                text = row[0]
                labels = int(row[1])
                train_data.append([text, labels])
train_df = pd.DataFrame(train_data, columns=['text', 'label'])

test_data = []
with open(f'{args.folder}'+'/test.tsv') as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            # Skip the header
            next(rd) 
            for row in rd:
                text = row[0]
                labels = int(row[1])
                test_data.append([text, labels])

test_df = pd.DataFrame(test_data, columns=['text', 'label'])


# Define ConfliBERT model as the learner for our AL framework
model = transformers.BertForSequenceClassification.from_pretrained(
    f'{args.pretrain}',
    num_labels=2,
    output_attentions=False,
    output_hidden_states=True,
)
tokenizer = transformers.BertTokenizer.from_pretrained(f'{args.pretrain}', max_length=512)
model.train()

# Preprocessing
encoder = OHE(sparse_output=False)
encoder.fit(train_df[['label']])

def tokenize(samples: 'list[str]'):
    return tokenizer(samples, return_tensors='pt', padding='max_length', truncation=True, max_length=512).input_ids

# Load test data
x_test = tokenize(test_df['text'].tolist()).to(device)

def test(device, apply_attention_mask, test_df, model, tokenizer, batch_size, annotation_round):
    preds = []

    with torch.no_grad():
        for i in range(0, x_test.shape[0], batch_size):
            x = x_test[i : i + batch_size].to(device)
            out = model(x, attention_mask=apply_attention_mask(x, tokenizer))
            preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())

# Calculate evaluation metrics
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


# Path to the file where you want to save the results
# Generate a timestamp string
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Define the result file name with the current timestamp
    result_file = f'{args.output}.txt'
    with open("Logs/"+result_file, 'a') as f:
    # Redirect print output to the file
        sys.stdout = f

    # Your print statements
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

    # Reset stdout to its default value (console)
        sys.stdout = sys.__stdout__

    print(f"Results saved to: {result_file}")

# Initial labelled dataset comprised of 1% of trainfing data
x_labeled, x_unlabeled = train_test_split(train_df, test_size=.99, stratify=train_df['label'])
x_train_labeled = tokenize(x_labeled['text'].tolist()).to(device)
x_train_unlabeled = tokenize(x_unlabeled['text'].tolist()).to(device)

y_train_labeled = torch.Tensor(encoder.transform(x_labeled[['label']])).to(device)
y_train_unlabeled  = torch.Tensor(encoder.transform(x_unlabeled [['label']])).to(device)

model.to(device)
x_train_labeled = x_train_labeled.to(device)
y_train_labeled = y_train_labeled.to(device)

class Perceptron(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Perceptron, self).__init__()
        self.dropout = nn.Dropout(0.9)
        self.fc = nn.Linear(input_size, num_classes)  # Adjust output size for your task

    def forward(self, x):
        x = self.dropout(x)
        return torch.softmax(self.fc(x), dim=1)  # Softmax activation for multi-class classification

def get_model(input_shape, num_classes):
    model = Perceptron(input_shape, num_classes)
    optimizer = AdamW(model.parameters(), lr=0.0001)
    return model, optimizer

p_model, optimizer  = get_model(768, 2)

# Active Learning loop
optim = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
num_samples_to_select = 10
num_annotation_rounds = 110
num_inference_cycles = 10
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

    cls_vector = []
    with torch.no_grad():
        for i in tqdm(range(0, x_train_labeled.shape[0], batch_size)):
            x = x_train_labeled[i : i + batch_size]
            outputs = model(x, attention_mask=apply_attention_mask(x, tokenizer))
            last_hidden_states = outputs.hidden_states[-1]
            cls_vector.extend(last_hidden_states[:, 0, :].to('cpu'))

    cls_vector = torch.stack(cls_vector)
    print(cls_vector.shape)

    print(f"Training MLP")
    for epoch in range(num_epochs):
        epoch_losses = []
        for i in range(0, cls_vector.shape[0], 10):
            p_model.train().to('cpu')
            x = cls_vector[i : i + 10]
            y = y_train_labeled[i : i + 10].to('cpu')
            optimizer.zero_grad()
            outputs = p_model(x)
            loss = nn.CrossEntropyLoss()(outputs, y)  
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())  # Append the loss of the current batch

        print(f'Losses at epoch {epoch}: {epoch_losses}')  # Print the losses after each epoch

   # Use the trained model to make predictions on the unlabeled set
    confidence_scores = []
    cls_vector_eval = []
    with torch.no_grad():
        model.eval()
        for i in range(0, x_train_unlabeled.shape[0], batch_size):
            x_batch_unlabeled = x_train_unlabeled[i : i + batch_size].to(device)
            model.zero_grad()
            outputs = model(x_batch_unlabeled, attention_mask=apply_attention_mask(x_batch_unlabeled, tokenizer))
            last_hidden_states = outputs.hidden_states[-1]
            cls_vector_eval.extend(last_hidden_states[:, 0, :].to('cpu'))

    entropies = []
    cls_vector_eval = torch.stack(cls_vector_eval)

    # Repeat inference 10 times
    for _ in range(num_inference_cycles):
        preds=[]
        with torch.no_grad():
            p_model.eval().to(device)
            p_model.dropout.train()
            for i in range(0, cls_vector_eval.shape[0], 10):
                x_batch_unlabeled = cls_vector_eval[i : i + 10].to(device)
                out = p_model(x_batch_unlabeled)
                preds.extend(out.to('cpu'))
        preds = torch.stack(preds)
        entropy = -torch.sum(preds * torch.log(preds), dim=1)  # Entropy formula
        entropies.append(entropy)

    print(len(entropies), entropies[0].shape)
    average_entropy = torch.mean(torch.stack(entropies), dim=0)
    print(average_entropy.shape)

    # Sort indices based on maximum entropy
    sorted_indices = torch.argsort(average_entropy, descending=True)

    selected_indices = sorted_indices[:num_samples_to_select]
    print(selected_indices)

   # Update labeled and unlabeled sets
    x_train_labeled = torch.cat([x_train_labeled, x_train_unlabeled[selected_indices].to(device)])
    y_train_labeled = torch.cat([y_train_labeled, y_train_unlabeled[selected_indices].to(device)])
    # Efficiently filter out least confident samples
    mask = torch.ones(x_train_unlabeled.shape[0], dtype=torch.bool, device=device)
    mask[selected_indices] = False
    x_train_unlabeled = x_train_unlabeled[mask]
    y_train_unlabeled = y_train_unlabeled[mask]
    if (annotation_round+1) % 5 == 0 :
        test(device, apply_attention_mask, test_df, model, tokenizer, batch_size, annotation_round+1)





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
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
import torch.nn as nn
from collections import Counter, defaultdict
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

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

def test(device, test_df, model, tokenizer, batch_size, annotation_round):
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

# Active Learning loop
optim = AdamW(model.parameters(), lr=1e-5, eps=1e-8)
num_samples_to_select = 50
num_annotation_rounds = 22
num_epochs = 10
batch_size = 32
num_inference_cycles = 10

def max_entropy_top_confidence_margin_sampling(device, model, x_train_unlabeled):
    confidence_scoresME = []
    confidence_scoresTC = []
    margin_scores = []
    for i in range(0, x_train_unlabeled.shape[0], batch_size):
        x_batch_unlabeled = x_train_unlabeled[i : i + batch_size].to(device)
        out = model(x_batch_unlabeled, attention_mask=apply_attention_mask(x_batch_unlabeled, tokenizer))
        logits = out.logits
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log2(probs + 1e-10), dim=1)
        confidence_scoresME.extend(entropy)
        top_two_probs, _ = torch.topk(probs, 2, dim=1)
        margins = top_two_probs[:, 0] - top_two_probs[:, 1]
        margin_scores.extend(margins.cpu().numpy())
        probsTC = torch.sigmoid(logits)
        confidence_scoresTC.extend(probsTC.max(dim=1).values.cpu())

    sorted_indices = sorted(range(len(confidence_scoresME)), key=lambda i: confidence_scoresME[i], reverse=True)
    selected_indices = sorted_indices[:num_samples_to_select]
    sorted_indicesTC = sorted(range(len(confidence_scoresTC)), key=lambda i: confidence_scoresTC[i])
    selected_indicesTC = sorted_indicesTC[:num_samples_to_select]
    sorted_indicesMS = sorted(range(len(margin_scores)), key=lambda i: margin_scores[i])
    selected_indicesMS = sorted_indicesMS[:num_samples_to_select]
    return selected_indices, selected_indicesTC, selected_indicesMS

def core_set(device, model, x_train_labeled, x_train_unlabeled, tokenizer, batch_size=16, num_samples_to_select=50):
    cls_vector_labeled = []
    cls_vector_unlabeled = []

    # ----- Labeled data embeddings -----
    with torch.no_grad():
        for i in range(0, x_train_labeled.shape[0], batch_size):
            x = x_train_labeled[i: i + batch_size].to(device)
            outputs = model(x, attention_mask=apply_attention_mask(x, tokenizer))
            hidden_states = outputs.hidden_states[-1]  # CLS token is at index 0
            cls_vector_labeled.extend(hidden_states[:, 0, :].cpu().numpy())

    # ----- Unlabeled data embeddings -----
    with torch.no_grad():
        for i in range(0, x_train_unlabeled.shape[0], batch_size):
            x = x_train_unlabeled[i: i + batch_size].to(device)
            outputs = model(x, attention_mask=apply_attention_mask(x, tokenizer))
            hidden_states = outputs.hidden_states[-1]
            cls_vector_unlabeled.extend(hidden_states[:, 0, :].cpu().numpy())

    # Convert lists → NumPy → Tensor (faster than list of ndarrays → tensor directly)
    cls_vector_labeled = torch.tensor(np.array(cls_vector_labeled), device=device)
    cls_vector_unlabeled = torch.tensor(np.array(cls_vector_unlabeled), device=device)

    # Track original indices for unlabeled pool
    orig_idx = torch.arange(cls_vector_unlabeled.size(0), device=device)

    # ----- Core-Set selection -----
    selected_indices = []
    for _ in range(num_samples_to_select):
        # Compute distances between unlabeled and labeled
        _, distances = pairwise_distances_argmin_min(
            cls_vector_unlabeled.cpu().numpy(), cls_vector_labeled.cpu().numpy()
        )
        max_distance_index = torch.argmax(torch.tensor(distances, device=device))

        # Map back to original index
        selected_indices.append(orig_idx[max_distance_index].item())

        # Move chosen sample from unlabeled → labeled pool
        cls_vector_labeled = torch.cat([cls_vector_labeled, cls_vector_unlabeled[max_distance_index].unsqueeze(0)])
        cls_vector_unlabeled = torch.cat([
            cls_vector_unlabeled[:max_distance_index],
            cls_vector_unlabeled[max_distance_index+1:]
        ])
        orig_idx = torch.cat([orig_idx[:max_distance_index], orig_idx[max_distance_index+1:]])

    return selected_indices

def monte_carlo(device, model, x_train_labeled, x_train_unlabeled, y_train_labeled):
    cls_vector = []
    p_model, optimizer  = get_model(768, 2)

    with torch.no_grad():
        for i in tqdm(range(0, x_train_labeled.shape[0], batch_size)):
            x = x_train_labeled[i : i + batch_size]
            outputs = model(x, attention_mask=apply_attention_mask(x, tokenizer))
            last_hidden_states = outputs.hidden_states[-1]
            cls_vector.extend(last_hidden_states[:, 0, :].to('cpu'))

    cls_vector = torch.stack(cls_vector)
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

    average_entropy = torch.mean(torch.stack(entropies), dim=0)

    # Sort indices based on maximum entropy
    sorted_indices = torch.argsort(average_entropy, descending=True)

    selected_indices = sorted_indices[:num_samples_to_select]
    return selected_indices

# Tokenize unlabeled data
# BALD Sampling requires DataLoader for batch processing
print("Tokenizing unlabeled data...")
# Tokenize function
def tokenize_texts(texts):
    return tokenizer(
        texts,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=512
    )
unlabeled_batch_size = 64  # Increased batch size for unlabeled data
tokenized_unlabeled = tokenize_texts(x_unlabeled['text'].tolist())
x_train_unlabeled_bald = tokenized_unlabeled['input_ids']
attention_mask_unlabeled = tokenized_unlabeled['attention_mask']

# Create indices for the unlabeled data
unlabeled_indices = np.arange(len(x_train_unlabeled_bald))

# Create unlabeled DataLoader with indices
unlabeled_dataset = TensorDataset(x_train_unlabeled_bald, attention_mask_unlabeled, torch.tensor(unlabeled_indices))
unlabeled_loader = DataLoader(
    unlabeled_dataset, batch_size=unlabeled_batch_size, shuffle=False, num_workers=4, pin_memory=True
)

def bald_sampling(device, model, unlabeled_tensor, tokenizer=None, num_samples_to_select=50, mc_iterations=10, batch_size=64, pad_token_id=0):
    """
    BALD (Bayesian Active Learning by Disagreement) sampling over input_ids tensor.
    Returns indices (descending MI) into unlabeled_tensor.
    """
    was_training = model.training  # remember incoming state
    model.train()                  # enable dropout
    all_scores = []

    with torch.no_grad():
        for start in range(0, unlabeled_tensor.shape[0], batch_size):
            x_batch = unlabeled_tensor[start:start+batch_size].to(device)
            attention_mask = (x_batch != pad_token_id).long().to(device)

            mc_probs = []
            for _ in range(mc_iterations):
                # ---- mixed precision forward pass ----
                with torch.cuda.amp.autocast():
                    out = model(input_ids=x_batch, attention_mask=attention_mask)
                    probs = F.softmax(out.logits, dim=-1)
                mc_probs.append(probs.unsqueeze(0))

            mc_probs = torch.cat(mc_probs, dim=0)     # [T, B, C]
            mean_probs = mc_probs.mean(dim=0)         # [B, C]

            # predictive entropy H[p(y|x,D)]
            pred_ent = -(mean_probs * (mean_probs + 1e-12).log()).sum(dim=-1)       # [B]

            # expected entropy E[ H[p(y|x,θ)] ]
            exp_ent = -(mc_probs * (mc_probs + 1e-12).log()).sum(dim=-1).mean(dim=0) # [B]

            mi = pred_ent - exp_ent  # BALD score
            all_scores.extend(mi.cpu().numpy())  # keep scores on CPU only at the end

    # restore original mode
    model.train(was_training)

    # highest MI first
    all_scores = np.asarray(all_scores)
    selected = np.argsort(-all_scores)[:num_samples_to_select]
    return selected

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

def calculate_union(lists_of_integers, top_n=50):
    position_scores = defaultdict(int)
    
    for lst in lists_of_integers:
        for idx, item in enumerate(lst):
            # Calculate the score based on the item's position in the list
            current_score = len(lst) - idx
            
            # Update the score only if this score is higher than the current score
            if current_score > position_scores[item]:
                position_scores[item] = current_score

    # Sort items by score in descending order and return as a list of tuples (item, score)
    sorted_scores = [item for item, score in sorted(position_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]]

    return sorted_scores


for annotation_round in range(num_annotation_rounds):
    selected_indices_count = 0
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
    selected_indices = []
    selected_indices_monte_carlo = monte_carlo(device, model, x_train_labeled, x_train_unlabeled, y_train_labeled)
    print(f"selected_indices after monte_carlo: {selected_indices_monte_carlo}")
    with torch.no_grad():
        model.eval()
        selected_indices_max_entropy, selected_indices_top_confidence,selected_indices_margin_sampling \
            = max_entropy_top_confidence_margin_sampling(device, model, x_train_unlabeled)
        print(f"selected_indices after max_entropy: {selected_indices_max_entropy}")
        #selected_indices_top_confidence = top_confidence(device, model, x_train_unlabeled)
        print(f"selected_indices after top_confidence: {selected_indices_top_confidence}")
        #selected_indices_margin_sampling = margin_sampling(device, model, x_train_unlabeled)
        print(f"selected_indices after margin_sampling: {selected_indices_margin_sampling}")
        selected_Bold_indices = bald_sampling(device, model, x_train_unlabeled)
        print(f"selected_indices after bald_sampling: {selected_Bold_indices}")
        selected_indices_core_set = core_set(device, model, x_train_labeled, x_train_unlabeled, tokenizer)
        print(f"selected_indices after core_set: {selected_indices_core_set}")
    


    # List comprehension
    lists_of_integers = [selected_indices_monte_carlo.view(-1).tolist(), selected_indices_max_entropy, selected_indices_top_confidence
    , selected_indices_margin_sampling, selected_indices_core_set, selected_Bold_indices]

    selected_indices = calculate_union(lists_of_integers) 

    selected_indices_count += len(selected_indices)
    print('Selected Indexes: ')
    for item in selected_indices:
        print(item)

    print(f"Adding {len(selected_indices)} labels")
   # Update labeled and unlabeled sets
    x_train_labeled = torch.cat([x_train_labeled, x_train_unlabeled[selected_indices].to(device)])
    y_train_labeled = torch.cat([y_train_labeled, y_train_unlabeled[selected_indices].to(device)])
    # Efficiently filter out least confident samples
    mask = torch.ones(x_train_unlabeled.shape[0], dtype=torch.bool, device=device)
    mask[selected_indices] = False
    x_train_unlabeled = x_train_unlabeled[mask]
    y_train_unlabeled = y_train_unlabeled[mask]
    test(device, test_df, model, tokenizer, batch_size, annotation_round+1)
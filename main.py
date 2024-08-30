import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import List, Tuple, Optional, Dict
from torchvision.models import resnet18
from torch.nn.utils import weight_norm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import pandas as pd
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import io
import os
import requests
import tarfile
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import json
import optuna

# Model definitions
class SelfModelingMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(SelfModelingMLP, self).__init__()
        self.hidden_layer = weight_norm(nn.Linear(input_dim, hidden_dim))
        self.output_layer = weight_norm(nn.Linear(hidden_dim, output_dim))
        self.self_model_layer = weight_norm(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x):
        hidden = torch.relu(self.hidden_layer(x))
        output = self.output_layer(hidden)
        self_model_output = self.self_model_layer(hidden)
        return output, self_model_output, hidden

class SelfModelingResNet(nn.Module):
    def __init__(self, num_classes: int, hidden_dim: int = 2000):
        super(SelfModelingResNet, self).__init__()
        self.resnet = resnet18(pretrained=False)
        self.resnet.fc = nn.Identity()  # Remove the final FC layer
        self.hidden_layer = weight_norm(nn.Linear(512, hidden_dim))
        self.output_layer = weight_norm(nn.Linear(hidden_dim, num_classes))
        self.self_model_layer = weight_norm(nn.Linear(hidden_dim, hidden_dim + 512))  # +512 for skip connection

    def forward(self, x):
        features = self.resnet(x)
        hidden = torch.relu(self.hidden_layer(features))
        output = self.output_layer(hidden)
        self_model_output = self.self_model_layer(hidden)
        return output, self_model_output, torch.cat([features, hidden], dim=1)

class SelfModelingIMDB(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, output_dim: int):
        super(SelfModelingIMDB, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.hidden_layer = weight_norm(nn.Linear(embed_dim, hidden_dim))
        self.output_layer = weight_norm(nn.Linear(hidden_dim, output_dim))
        self.self_model_layer = weight_norm(nn.Linear(hidden_dim, hidden_dim))

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        hidden = torch.relu(self.hidden_layer(embedded))
        output = self.output_layer(hidden)
        self_model_output = self.self_model_layer(hidden)
        return output, self_model_output, hidden

class SelfModelingLoss(nn.Module):
    def __init__(self, task_loss: nn.Module, self_model_weight: float):
        super(SelfModelingLoss, self).__init__()
        self.task_loss = task_loss
        self.self_model_weight = self_model_weight

    def forward(self, outputs, targets, self_model_outputs, activations):
        task_loss = self.task_loss(outputs, targets)
        self_model_loss = nn.MSELoss()(self_model_outputs, activations.detach())
        return task_loss + self.self_model_weight * self_model_loss

# Training and evaluation functions
def train_epoch(model, dataloader, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss, correct = 0, 0
    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, self_model_outputs, activations = model(inputs)
        loss = criterion(outputs, targets, self_model_outputs, activations)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(1) == targets).sum().item()
    return total_loss / len(dataloader.dataset), correct / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, self_model_outputs, activations = model(inputs)
            loss = criterion(outputs, targets, self_model_outputs, activations)
            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
    return total_loss / len(dataloader.dataset), correct / len(dataloader.dataset)

# Complexity measures
def get_weight_distribution(model):
    return np.concatenate([p.detach().cpu().numpy().flatten() for p in model.parameters()])

def estimate_rlct(model, dataloader, criterion, device, lr=0.0001, localization=1000, num_samples=1000):
    model.eval()
    original_params = [p.clone().detach() for p in model.parameters()]
    
    losses = []
    for _ in tqdm(range(num_samples), desc="Estimating RLCT", leave=False):
        for p, orig_p in zip(model.parameters(), original_params):
            noise = torch.randn_like(p) * np.sqrt(2 * lr)
            p.data.add_(noise).add_(-(p.data - orig_p) * localization * lr)
        
        batch_losses = []
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, self_model_outputs, activations = model(inputs)
            loss = criterion(outputs, targets, self_model_outputs, activations)
            batch_losses.append(loss.item())
        losses.append(np.mean(batch_losses))
    
    # Restore original parameters
    for p, orig_p in zip(model.parameters(), original_params):
        p.data.copy_(orig_p)
    
    # Estimate RLCT
    log_losses = np.log(losses)
    rlct = np.mean(log_losses) / 2
    return rlct

# IMDB dataset handling
def download_imdb_dataset(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.raw.read())

def extract_imdb_dataset(tar_path, extract_path):
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)

def load_imdb_data(data_path):
    data = []
    for sentiment in ['pos', 'neg']:
        sentiment_path = os.path.join(data_path, sentiment)
        for filename in os.listdir(sentiment_path):
            with open(os.path.join(sentiment_path, filename), 'r', encoding='utf-8') as file:
                data.append((file.read(), 1 if sentiment == 'pos' else 0))
    return data

class IMDBDataset(Dataset):
    def __init__(self, data, tokenizer, vocab, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        tokens = self.tokenizer(text)
        ids = [self.vocab[token] for token in tokens]
        if len(ids) > self.max_length:
            ids = ids[:self.max_length]
        return torch.tensor(ids), torch.tensor(label)

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_text, _label) in batch:
        label_list.append(_label)
        processed_text = torch.tensor(_text, dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return text_list, label_list, offsets

# Experiment runner
def run_experiment(model, train_loader, test_loader, optimizer, criterion, device, num_epochs, scheduler=None, writer=None):
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/test', test_acc, epoch)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # Compute complexity measures
    weight_dist = get_weight_distribution(model)
    rlct = estimate_rlct(model, test_loader, criterion, device)
    
    if writer:
        writer.add_histogram('Weight Distribution', weight_dist, 0)
        writer.add_scalar('RLCT', rlct, 0)
    
    return train_loss, train_acc, test_loss, test_acc, weight_dist, rlct

# Task-specific experiment functions
def run_mnist_experiment(hidden_dim, self_model_weight, num_epochs, writer=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model = SelfModelingMLP(784, hidden_dim, 10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, nesterov=True)
    criterion = SelfModelingLoss(nn.CrossEntropyLoss(), self_model_weight)
    
    return run_experiment(model, train_loader, test_loader, optimizer, criterion, device, num_epochs, writer=writer)

def run_cifar10_experiment(self_model_weight, num_epochs, writer=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model = SelfModelingResNet(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = SelfModelingLoss(nn.CrossEntropyLoss(), self_model_weight)
    
    return run_experiment(model, train_loader, test_loader, optimizer, criterion, device, num_epochs, writer=writer)

def run_imdb_experiment(self_model_weight, num_epochs, writer=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Download and extract IMDB dataset
    url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    save_path = "aclImdb_v1.tar.gz"
    extract_path = "."
    
    if not os.path.exists("aclImdb"):
        print("Downloading IMDB dataset...")
        download_imdb_dataset(url, save_path)
        print("Extracting IMDB dataset...")
        extract_imdb_dataset(save_path, extract_path)
    
    # Load and preprocess data
    print("Loading IMDB data...")
    data = load_imdb_data("aclImdb/train") + load_imdb_data("aclImdb/test")
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab_from_iterator(map(lambda x: tokenizer(x[0]), train_data), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    
    max_length = 256
    train_dataset = IMDBDataset(train_data, tokenizer, vocab, max_length)
    test_dataset = IMDBDataset(test_data, tokenizer, vocab, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_batch)
    
    vocab_size = len(vocab)
    embed_dim = 64
    hidden_dim = 128
    model = SelfModelingIMDB(vocab_size, embed_dim, hidden_dim, 2).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = SelfModelingLoss(nn.CrossEntropyLoss(), self_model_weight)
    
    return run_experiment(model, train_loader, test_loader, optimizer, criterion, device, num_epochs, writer=writer)

# Hyperparameter tuning
def objective(trial, task):
    if task == 'mnist':
        hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
        self_model_weight = trial.suggest_loguniform('self_model_weight', 1e-3, 1e2)
        num_epochs = 50
        _, _, _, test_acc, _, rlct = run_mnist_experiment(hidden_dim, self_model_weight, num_epochs)
    elif task == 'cifar10':
        self_model_weight = trial.suggest_loguniform('self_model_weight', 1e-3, 1e2)
        num_epochs = 250
        _, _, _, test_acc, _, rlct = run_cifar10_experiment(self_model_weight, num_epochs)
    elif task == 'imdb':
        self_model_weight = trial.suggest_loguniform('self_model_weight', 1e-3, 1e2)
        num_epochs = 500
        _, _, _, test_acc, _, rlct = run_imdb_experiment(self_model_weight, num_epochs)
    else:
        raise ValueError(f"Unknown task: {task}")
    
    return test_acc, rlct

def run_hyperparameter_tuning(task, n_trials=100):
    study = optuna.create_study(directions=['maximize', 'minimize'])
    study.optimize(lambda trial: objective(trial, task), n_trials=n_trials)
    
    print("Best trials:")
    best_trials = study.best_trials
    for trial in best_trials:
        print(f"  Value (accuracy, RLCT): {trial.values}")
        print(f"  Params: {trial.params}")

# Visualization functions
def plot_weight_distribution(weight_dist, title):
    plt.figure(figsize=(10, 6))
    plt.hist(weight_dist, bins=50, density=True)
    plt.title(title)
    plt.xlabel('Weight Value')
    plt.ylabel('Density')
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.close()

def plot_rlct_vs_self_model_weight(results, task):
    plt.figure(figsize=(10, 6))
    weights = [r['self_model_weight'] for r in results]
    rlcts = [r['rlct'] for r in results]
    plt.semilogx(weights, rlcts, 'o-')
    plt.title(f'RLCT vs Self-Model Weight for {task.upper()}')
    plt.xlabel('Self-Model Weight')
    plt.ylabel('RLCT')
    plt.savefig(f"rlct_vs_weight_{task}.png")
    plt.close()

# Main execution
if __name__ == "__main__":
    tasks = ['mnist', 'cifar10', 'imdb']
    
    for task in tasks:
        print(f"Running hyperparameter tuning for {task}")
        run_hyperparameter_tuning(task)
    
    # Run experiments with best hyperparameters and visualize results
    for task in tasks:
        results = []
        if task == 'mnist':
            hidden_dims = [64, 128, 256, 512]
            self_model_weights = [1, 5, 10, 20, 50]
            for hidden_dim in hidden_dims:
                for weight in self_model_weights:
                    writer = SummaryWriter(f'runs/{task}_hd{hidden_dim}_w{weight}')
                    train_loss, train_acc, test_loss, test_acc, weight_dist, rlct = run_mnist_experiment(hidden_dim, weight, 50, writer)
                    results.append({
                        'hidden_dim': hidden_dim,
                        'self_model_weight': weight,
                        'test_acc': test_acc,
                        'rlct': rlct,
                        'weight_dist': weight_dist
                    })
                    writer.close()
                    plot_weight_distribution(weight_dist, f'MNIST Weight Distribution (HD={hidden_dim}, W={weight})')
        elif task == 'cifar10':
            self_model_weights = [0.5, 1, 2]
            for weight in self_model_weights:
                writer = SummaryWriter(f'runs/{task}_w{weight}')
                train_loss, train_acc, test_loss, test_acc, weight_dist, rlct = run_cifar10_experiment(weight, 250, writer)
                results.append({
                    'self_model_weight': weight,
                    'test_acc': test_acc,
                    'rlct': rlct,
                    'weight_dist': weight_dist
                })
                writer.close()
                plot_weight_distribution(weight_dist, f'CIFAR-10 Weight Distribution (W={weight})')
        elif task == 'imdb':
            self_model_weights = [100, 500]
            for weight in self_model_weights:
                writer = SummaryWriter(f'runs/{task}_w{weight}')
                train_loss, train_acc, test_loss, test_acc, weight_dist, rlct = run_imdb_experiment(weight, 500, writer)
                results.append({
                    'self_model_weight': weight,
                    'test_acc': test_acc,
                    'rlct': rlct,
                    'weight_dist': weight_dist
                })
                writer.close()
                plot_weight_distribution(weight_dist, f'IMDB Weight Distribution (W={weight})')
        
        plot_rlct_vs_self_model_weight(results, task)
        
        # Save results to JSON
        with open(f'{task}_results.json', 'w') as f:
            json.dump(results, f, cls=NumpyEncoder)

# Custom JSON encoder to handle numpy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
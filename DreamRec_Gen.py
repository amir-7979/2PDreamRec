import os
import time as Time
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm import tqdm
import random

from Modules_ori import MovieDiffusion, Tenc, MovieTenc, load_genres_predictor, diffusion, Tenc
from utility import extract_axis_1, calculate_hit

import logging

logging.getLogger().setLevel(logging.INFO)

############################################
# Directory for Merged Data Splits
############################################
# All merged CSV files (for training, validation, and test) reside in this directory.
MERGED_DATA_DIR = "data"


############################################
# Dataset Definition (Merged for Genre)
############################################
class GenreDataset(Dataset):
    """
    A PyTorch dataset for genre prediction using merged data.
    Expected CSV columns:
      - genre_seq: string representation of a list of genre IDs
      - genre_len: integer (length of genre sequence; if missing, it is computed)
      - genre_target: target genre ID
    """

    def __init__(self, dataframe):
        self.genre_seq = dataframe['genre_seq'].tolist()
        if 'genre_len' in dataframe.columns:
            self.genre_len = dataframe['genre_len'].tolist()
        else:
            self.genre_len = [len(eval(s)) for s in self.genre_seq]
        self.genre_target = dataframe['genre_target'].tolist()

    def __len__(self):
        return len(self.genre_seq)

    def __getitem__(self, idx):
        return (torch.tensor(eval(self.genre_seq[idx]), dtype=torch.long),
                torch.tensor(self.genre_len[idx], dtype=torch.long),
                torch.tensor(self.genre_target[idx], dtype=torch.long))


############################################
# Argument Parsing and Setup
############################################
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Genre Prediction with Diffusion + Cross-Entropy & Focal Loss using 10-Fold CV on merged data."
    )
    parser.add_argument('--tune', action='store_true', default=False, help='Enable tuning.')
    parser.add_argument('--no-tune', action='store_false', dest='tune', help='Disable tuning.')
    parser.add_argument('--epoch', type=int, default=100, help='Number of epochs per fold.')
    parser.add_argument('--random_seed', type=int, default=100, help='Random seed.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64, help='Embedding/hidden size.')
    parser.add_argument('--timesteps', type=int, default=100, help='Timesteps for diffusion.')
    parser.add_argument('--beta_end', type=float, default=0.02, help='Beta end of diffusion.')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='Beta start of diffusion.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--l2_decay', type=float, default=0.3, help='L2 loss regularization coefficient.')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device id.')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate.')
    parser.add_argument('--w', type=float, default=2.0, help='Weight used in x_start update inside sampler.')
    parser.add_argument('--p', type=float, default=0.1, help='Probability used in cacu_h for random dropout.')
    parser.add_argument('--report_epoch', type=bool, default=True, help='Whether to report metrics each epoch.')
    parser.add_argument('--diffuser_type', type=str, default='mlp1', help='Type of diffuser network: [mlp1, mlp2].')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer type: [adam, adamw, adagrad, rmsprop].')
    parser.add_argument('--beta_sche', nargs='?', default='linear', help='Beta schedule: [linear, exp, cosine, sqrt].')
    parser.add_argument('--descri', type=str, default='', help='Description of the run.')
    return parser.parse_args()


args = parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(args.random_seed)

device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


############################################
# Hyperparameter Metric Class (for tuning)
############################################
class Metric:
    def __init__(self, name, values):
        self.name = name
        self.values = values
        self.eval_dict = defaultdict(list)
        self.bestOne = None

    def find_max_one(self):
        best = -np.inf
        for key in self.eval_dict.keys():
            temp = max(self.eval_dict[key])
            if temp > best:
                self.bestOne = key
                best = temp

    def find_min_one(self):
        best = np.inf
        for key in self.eval_dict.keys():
            temp = min(self.eval_dict[key])
            if temp < best:
                self.bestOne = key
                best = temp


############################################
# Evaluate Function (Merged Data for Genre)
############################################
def evaluate(model, diff, dataset_split, device):
    """
    Evaluate the genre prediction model on a given split.
    Loads merged CSV from MERGED_DATA_DIR (with columns: genre_seq, genre_len, genre_target)
    and computes evaluation metrics.
    """
    eval_data = pd.read_csv(os.path.join(MERGED_DATA_DIR, dataset_split))
    batch_size = args.batch_size
    topk = [5, 10]
    total_samples = 0
    hit_purchase = np.zeros(len(topk))
    ndcg_purchase = np.zeros(len(topk))
    losses = []

    genre_seq = eval_data['genre_seq'].apply(lambda x: list(map(int, eval(x)))).tolist()
    genre_len = eval_data['genre_len'].values if 'genre_len' in eval_data.columns else np.array(
        [len(eval(x)) for x in eval_data['genre_seq'].tolist()])
    genre_target = eval_data['genre_target'].values

    for i in range(0, len(genre_seq), batch_size):
        seq_batch = torch.LongTensor(genre_seq[i:i + batch_size]).to(device)
        len_seq_batch = torch.LongTensor(genre_len[i:i + batch_size]).to(device)
        target_batch = torch.LongTensor(genre_target[i:i + batch_size]).to(device)

        x_start = model.cacu_x(target_batch)
        h = model.cacu_h(seq_batch, len_seq_batch, args.p)
        n = torch.randint(0, args.timesteps, (h.size(0),), device=device).long()

        loss1, predicted_x = diff.p_losses(model, x_start, h, n, loss_type='l2')
        seq_batch_tensor = torch.tensor(genre_seq, dtype=torch.long, device=device)
        prediction = model.predict(seq_batch_tensor, len_seq_batch, diff)

        losses.append(loss1.item())
        _, topK = prediction.topk(10, dim=1, largest=True, sorted=True)
        topK = topK.cpu().detach().numpy()
        calculate_hit(target_batch, topK, topk, hit_purchase, ndcg_purchase)
        total_samples += len(target_batch)

    avg_loss = sum(losses) / len(losses)
    hr_list = hit_purchase / total_samples
    ndcg_list = ndcg_purchase / total_samples
    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format(
        'HR@' + str(topk[0]), 'NDCG@' + str(topk[0]),
        'HR@' + str(topk[1]), 'NDCG@' + str(topk[1])))
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(
        hr_list[0], ndcg_list[0], hr_list[1], ndcg_list[1]))
    print(f'Loss: {avg_loss:.4f}')
    return avg_loss, hr_list[0]

############################################
# Training Function for One Fold (Merged Data for Genre)
############################################
def train_fold(fold):
    print(f"\n========== Fold {fold} ==========")
    # Filenames for merged data splits
    train_csv = f"train_fold{fold}.df"
    val_csv = f"val_fold{fold}.df"
    test_csv = f"test_fold{fold}.df"

    train_df = pd.read_csv(os.path.join(MERGED_DATA_DIR, train_csv))
    val_df = pd.read_csv(os.path.join(MERGED_DATA_DIR, val_csv))
    test_df = pd.read_csv(os.path.join(MERGED_DATA_DIR, test_csv))

    train_dataset = GenreDataset(train_df)
    val_dataset = GenreDataset(val_df)
    test_dataset = GenreDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Data statistics for genre prediction
    seq_size = 10  # maximum sequence length for genres
    # Load the genre mapping CSV (if it exists)
    genre_mapping_path = os.path.join(MERGED_DATA_DIR, "genre_mapping.csv")
    if os.path.exists(genre_mapping_path):
        genre_mapping_df = pd.read_csv(genre_mapping_path)
        genre_vocab_size = genre_mapping_df.shape[0]
    else:
    # Fallback: if no mapping file exists, set a default (or compute from your dataset)
        genre_vocab_size = 18  # or compute dynamically from your training data

    # Initialize the base encoder (Tenc) and attach a simple genre decoder.
    model = Tenc(args.hidden_factor, genre_vocab_size, seq_size, args.dropout_rate, args.diffuser_type, device)
    # Attach the genre decoder

    # Use the base diffusion class for genre prediction.
    diff = diffusion(args.timesteps, args.beta_start, args.beta_end, args.w)

    # (Optionally, load a pre-trained genre model here.)
    # For now, we train from scratch.
    model.to(device)
    # If diffusion has parameters, move them to device (if not, ignore)
    # diff.to(device)  # Uncomment if your diffusion object has learnable parameters.

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-3, weight_decay=args.l2_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, eps=1e-3, weight_decay=args.l2_decay)
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr, eps=1e-3, weight_decay=args.l2_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, eps=1e-3, weight_decay=args.l2_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-6, weight_decay=args.l2_decay)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    # Training loop
    for epoch in range(args.epoch):
        start_time = Time.time()
        model.train()
        for batch_data in train_loader:
            genre_seq_batch, genre_len_batch, genre_target_batch = batch_data
            genre_seq_batch = genre_seq_batch.to(device)
            genre_len_batch = genre_len_batch.to(device)
            genre_target_batch = genre_target_batch.to(device)
            optimizer.zero_grad()
            x_start = model.cacu_x(genre_target_batch)
            n = torch.randint(0, args.timesteps, (genre_seq_batch.shape[0],), device=device).long()
            h = model.cacu_h(genre_seq_batch, genre_len_batch, args.p)
            loss1, predicted_x = diff.p_losses(model, x_start, h, n, loss_type='l2')

            loss = loss1
            loss.backward()
            optimizer.step()
        if args.report_epoch:
            print(
                f"Fold {fold} Epoch {epoch:03d}; Train loss: {loss.item():.4f}; Time: {Time.strftime('%H:%M:%S', Time.gmtime(Time.time() - start_time))}")
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                print(f"Fold {fold}: Validation Phase")
                avg_loss, hr_val = evaluate(model, diff, val_csv, device)
                print(f"Fold {fold}: Val Loss: {avg_loss:.4f}, HR@5: {hr_val:.4f}")
            scheduler.step()

    # After training, evaluate on the test set:
    with torch.no_grad():
        print(f"Fold {fold}: Test Phase")
        avg_loss_test, hr_test = evaluate(model, diff, test_csv, device)
        print(f"Fold {fold}: Test Loss: {avg_loss_test:.4f}, HR@5: {hr_test:.4f}")

    os.makedirs("./models", exist_ok=True)
    torch.save(model.state_dict(), f"./models/genre_tenc_fold{fold}.pth")
    torch.save(diff, f"./models/genre_diff_fold{fold}.pth")

    return hr_val, hr_test


############################################
# Main Function
############################################
def main():
    NUM_FOLDS = 10
    fold_val_metrics = []
    fold_test_metrics = []
    if args.tune:
        metrics = [
            Metric(name='lr', values=[0.1, 0.01, 0.001, 0.0001, 0.00001]),
            Metric(name='optimizer', values=['adam', 'adamw', 'adagrad', 'rmsprop']),
            Metric(name='timesteps', values=[i * 50 for i in range(1, 11)]),
        ]
        best_metrics = []
        for metric in metrics:
            for value in tqdm(metric.values, desc=f"Tuning {metric.name}"):
                if metric.name == 'lr':
                    args.lr = value
                elif metric.name == 'optimizer':
                    args.optimizer = value
                elif metric.name == 'timesteps':
                    args.timesteps = value
                print(f"Setting {metric.name} = {value}")
                hr_val_sum = 0
                hr_test_sum = 0
                for fold in range(1, NUM_FOLDS + 1):
                    hr_val, hr_test = train_fold(fold)
                    hr_val_sum += hr_val
                    hr_test_sum += hr_test
                hr_val_avg = hr_val_sum / NUM_FOLDS
                hr_test_avg = hr_test_sum / NUM_FOLDS
                print(
                    f"Hyperparameter {metric.name} = {value} gives average Val HR@5: {hr_val_avg:.4f}, Test HR@5: {hr_test_avg:.4f}")
                metric.eval_dict[value].append(hr_val_avg)
            metric.find_max_one()
            best_metrics.append(metric)
        os.makedirs("./tune", exist_ok=True)
        torch.save(best_metrics, './tune/metrics.dict')
        print("Tuning complete. Best metrics:")
        for m in best_metrics:
            print(f"{m.name}: {m.bestOne}")
    else:
        for fold in range(1, NUM_FOLDS + 1):
            hr_val, hr_test = train_fold(fold)
            fold_val_metrics.append(hr_val)
            fold_test_metrics.append(hr_test)
        print("\n========== 10-Fold Cross Validation Complete ==========")
        print("Average Val HR@5 across folds: {:.4f} ± {:.4f}".format(np.mean(fold_val_metrics),
                                                                      np.std(fold_val_metrics)))
        print("Average Test HR@5 across folds: {:.4f} ± {:.4f}".format(np.mean(fold_test_metrics),
                                                                       np.std(fold_test_metrics)))


if __name__ == '__main__':
    main()

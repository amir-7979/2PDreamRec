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
# Evaluation Function
############################################
def evaluate(model, diff, dataset_split, device):
    # Set a fixed random seed for evaluation to ensure determinism.
    fixed_seed = 100
    torch.manual_seed(fixed_seed)
    np.random.seed(fixed_seed)
    random.seed(fixed_seed)

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
        seq_batch = torch.LongTensor(genre_seq[i:i+batch_size]).to(device)
        len_seq_batch = torch.LongTensor(genre_len[i:i+batch_size]).to(device)
        target_batch = torch.LongTensor(genre_target[i:i+batch_size]).to(device)

        x_start = model.cacu_x(target_batch)
        h = model.cacu_h(seq_batch, len_seq_batch, args.p)
        n = torch.randint(0, args.timesteps, (h.size(0),), device=device).long()
        loss1, predicted_x = diff.p_losses(model, x_start, h, n, loss_type='l2')
        losses.append(loss1.item())

        prediction = model.predict(seq_batch, len_seq_batch, diff)
        _, topK = prediction.topk(10, dim=1, largest=True, sorted=True)
        topK = topK.cpu().detach().numpy()
        calculate_hit(target_batch, topK, topk, hit_purchase, ndcg_purchase)
        total_samples += len(target_batch)

    avg_loss = np.mean(losses)
    hr_list = hit_purchase / total_samples
    ndcg_list = ndcg_purchase / total_samples

    return {'loss': avg_loss, 'HR5': hr_list[0], 'NDCG5': ndcg_list[0],
            'HR10': hr_list[1], 'NDCG10': ndcg_list[1]}

############################################
# Classes for Recording Metrics
############################################
class FoldMetrics:
    def __init__(self, fold_number):
        self.fold_number = fold_number
        self.train_losses = []  # list of tuples: (epoch, loss)
        self.val_metrics = {}   # dict: epoch -> metrics dict
        self.test_metrics = {}  # dict: epoch -> metrics dict

    def add_train_loss(self, epoch, loss):
        self.train_losses.append((epoch, loss))

    def add_val_metrics(self, epoch, metrics):
        self.val_metrics[epoch] = metrics

    def add_test_metrics(self, epoch, metrics):
        self.test_metrics[epoch] = metrics

    def __str__(self):
        s = f"Fold {self.fold_number} Metrics:\n"
        s += "Epoch\tTrainLoss\tValLoss\tTestLoss\tHR@5\tNDCG@5\tHR@10\tNDCG@10\n"
        for epoch in sorted(self.val_metrics.keys()):
            train_loss = next((tl for ep, tl in self.train_losses if ep == epoch), None)
            val = self.val_metrics[epoch]
            test = self.test_metrics.get(epoch, None)
            test_loss = test['loss'] if test is not None else float('nan')
            s += f"{epoch}\t{train_loss:.4f}\t{val['loss']:.4f}\t{test_loss:.4f}\t{test['HR5']:.4f}\t{test['NDCG5']:.4f}\t{test['HR10']:.4f}\t{test['NDCG10']:.4f}\n"
        return s

class AverageMetrics:
    def __init__(self):
        self.avg_train_loss = {}
        self.avg_val_metrics = {}
        self.avg_test_metrics = {}
        self.num_folds = 0

    def add_fold_metrics(self, fold_metric):
        self.num_folds += 1
        for epoch, loss in fold_metric.train_losses:
            self.avg_train_loss.setdefault(epoch, []).append(loss)
        for epoch, metrics in fold_metric.val_metrics.items():
            self.avg_val_metrics.setdefault(epoch, []).append(metrics)
        for epoch, metrics in fold_metric.test_metrics.items():
            self.avg_test_metrics.setdefault(epoch, []).append(metrics)

    def compute_averages(self):
        self.avg_train_loss = {epoch: np.mean(losses) for epoch, losses in self.avg_train_loss.items()}
        def avg_dict(metrics_list):
            avg_d = {}
            for key in metrics_list[0].keys():
                avg_d[key] = np.mean([m[key] for m in metrics_list])
            return avg_d
        self.avg_val_metrics = {epoch: avg_dict(metrics_list) for epoch, metrics_list in self.avg_val_metrics.items()}
        self.avg_test_metrics = {epoch: avg_dict(metrics_list) for epoch, metrics_list in self.avg_test_metrics.items()}

    def __str__(self):
        s = "Average Metrics Across Folds:\n"
        s += "Epoch\tAvgTrainLoss\tAvgValLoss\tAvgTestLoss\tAvgHR@5\tAvgNDCG@5\tAvgHR@10\tAvgNDCG@10\n"
        for epoch in sorted(self.avg_val_metrics.keys()):
            train_loss = self.avg_train_loss.get(epoch, None)
            val = self.avg_val_metrics[epoch]
            test = self.avg_test_metrics.get(epoch, None)
            test_loss = test['loss'] if test is not None else float('nan')
            s += (f"{epoch}\t{train_loss:.4f}\t{val['loss']:.4f}\t{test_loss:.4f}\t{test['HR5']:.4f}\t"
                  f"{test['NDCG5']:.4f}\t{test['HR10']:.4f}\t{test['NDCG10']:.4f}\n")
        return s

############################################
# Modified Tuning Classes (Recording lists per candidate)
############################################
class TuningMetric:
    """
    Records, for each candidate, the list of HR@10 values at each evaluation checkpoint (e.g. epochs 10,20,...,100)
    from each fold.
    """
    def __init__(self, name, values):
        self.name = name            # e.g., 'lr'
        self.values = values        # candidate values list
        # For each candidate, use a dict: key = evaluation epoch, value = list of HR@10 values (from each fold)
        self.eval_dict = defaultdict(lambda: defaultdict(list))
        self.best_value = None

    def record(self, candidate, fold_hr10_list):
        """
        fold_hr10_list: a list of HR@10 values (one per evaluation checkpoint)
        """
        for i, hr in enumerate(fold_hr10_list):
            # Evaluation checkpoint epoch = (i+1)*10 (assuming evaluations occur every 10 epochs)
            self.eval_dict[candidate][(i+1)*10].append(hr)

    def average(self):
        """
        For each candidate, compute the average HR@10 at each evaluation epoch.
        Returns a dict: candidate -> list of average HR@10 at epochs 10,20,...,100.
        """
        self.avg_dict = {}
        for candidate in self.values:
            avg_list = []
            for epoch in sorted(self.eval_dict[candidate].keys()):
                avg_list.append(np.mean(self.eval_dict[candidate][epoch]))
            self.avg_dict[candidate] = avg_list
        return self.avg_dict

    def find_best(self):
        """
        Choose the best candidate based on the average HR@10 at the final evaluation checkpoint (epoch 100).
        """
        self.average()
        best = -np.inf
        for candidate in self.values:
            avg_final = np.mean(self.eval_dict[candidate][100])
            if avg_final > best:
                best = avg_final
                self.best_value = candidate
        return self.best_value

    def __str__(self):
        s = f"TuningMetric for {self.name}:\n"
        for candidate in self.values:
            avg_list = self.avg_dict.get(candidate, None)
            s += f"Candidate: {candidate}, HR@10 over eval epochs: {avg_list}\n"
        s += f"Best candidate: {self.best_value}\n"
        return s

class TuningSummary:
    def __init__(self):
        self.metrics = {}
    def add_metric(self, tuning_metric: TuningMetric):
        self.metrics[tuning_metric.name] = tuning_metric
    def __str__(self):
        s = "Tuning Summary:\n"
        for metric in self.metrics.values():
            s += str(metric) + "\n"
        return s

############################################
# Training Function for One Fold (unchanged)
############################################
def train_fold(fold):
    print(f"\n========== Fold {fold} ==========")
    fold_metrics = FoldMetrics(fold)
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

    # Set model parameters from statics if available.
    statics_path = os.path.join(MERGED_DATA_DIR, "statics.csv")
    if os.path.exists(statics_path):
        statics_df = pd.read_csv(statics_path)
        statics = dict(zip(statics_df['statistic'], statics_df['value']))
        seq_size = int(statics.get("train_seq_length", 10))
        genre_vocab_size = int(statics.get("num_genres", 18))
        logging.info(f"Using statics: seq_size = {seq_size}, genre_vocab_size = {genre_vocab_size}")
    else:
        seq_size = 10
        genre_mapping_path = os.path.join(MERGED_DATA_DIR, "genre_mapping.csv")
        if os.path.exists(genre_mapping_path):
            genre_mapping_df = pd.read_csv(genre_mapping_path)
            genre_vocab_size = genre_mapping_df.shape[0]
        else:
            genre_vocab_size = 18

    model = Tenc(args.hidden_factor, genre_vocab_size, seq_size, args.dropout_rate, args.diffuser_type, device)
    diff = diffusion(args.timesteps, args.beta_start, args.beta_end, args.w)
    model.to(device)

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-6, weight_decay=args.l2_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, eps=1e-6, weight_decay=args.l2_decay)
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr, eps=1e-6, weight_decay=args.l2_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, eps=1e-6, weight_decay=args.l2_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-6, weight_decay=args.l2_decay)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    # Training loop: record train loss each epoch; every 10 epochs, evaluate.
    for epoch in range(args.epoch):
        start_time = Time.time()
        model.train()
        epoch_losses = []
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
            epoch_losses.append(loss.item())
        avg_epoch_loss = np.mean(epoch_losses)
        fold_metrics.add_train_loss(epoch + 1, avg_epoch_loss)
        if args.report_epoch:
            print(f"Fold {fold} Epoch {epoch + 1:03d}; Train loss: {avg_epoch_loss:.4f}; Time: {Time.strftime('%H:%M:%S', Time.gmtime(Time.time() - start_time))}")
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                print(f"Fold {fold}: Evaluation at Epoch {epoch + 1}")
                train_eval_met = evaluate(model, diff, train_csv, device)
                val_met = evaluate(model, diff, val_csv, device)
                test_met = evaluate(model, diff, test_csv, device)
            fold_metrics.add_train_loss(epoch + 1, train_eval_met['loss'])
            fold_metrics.add_val_metrics(epoch + 1, val_met)
            fold_metrics.add_test_metrics(epoch + 1, test_met)
            scheduler.step()

    os.makedirs("./models", exist_ok=True)
    torch.save(model.state_dict(), f"./models/genre_tenc_fold{fold}.pth")
    torch.save(diff, f"./models/genre_diff_fold{fold}.pth")
    return fold_metrics

############################################
# Main Function with Tuning (Using Only Fold 1)
############################################
def main():
    # For tuning, we use only fold 1 for faster experiments.
    tuning_fold = 1

    # Define candidate values.
    lr_candidates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    optimizer_candidates = ['adam', 'adamw', 'adagrad', 'rmsprop']
    timesteps_candidates = [i * 100 for i in range(1, 11)]

    # Create tuning metric objects.
    tuning_lr = TuningMetric("lr", lr_candidates)
    tuning_optimizer = TuningMetric("optimizer", optimizer_candidates)
    tuning_timesteps = TuningMetric("timesteps", timesteps_candidates)

    # --- Tuning learning rate using only fold 1 ---
    for candidate in tqdm(lr_candidates, desc="Tuning lr"):
        args.lr = candidate
        fm = train_fold(tuning_fold)
        # Extract HR@10 values from each evaluation checkpoint (epochs 10,20,...,100)
        fold_hr10_list = [fm.test_metrics[epoch]['HR10'] for epoch in sorted(fm.test_metrics.keys())]
        tuning_lr.record(candidate, fold_hr10_list)
        print(f"[lr candidate {candidate}] Fold {tuning_fold}: HR@10 = {fold_hr10_list}")

    # --- Tuning optimizer using only fold 1 ---
    for candidate in tqdm(optimizer_candidates, desc="Tuning optimizer"):
        args.optimizer = candidate
        fm = train_fold(tuning_fold)
        fold_hr10_list = [fm.test_metrics[epoch]['HR10'] for epoch in sorted(fm.test_metrics.keys())]
        tuning_optimizer.record(candidate, fold_hr10_list)
        print(f"[optimizer candidate {candidate}] Fold {tuning_fold}: HR@10 = {fold_hr10_list}")

    # --- Tuning timesteps using only fold 1 ---
    for candidate in tqdm(timesteps_candidates, desc="Tuning timesteps"):
        args.timesteps = candidate
        fm = train_fold(tuning_fold)
        fold_hr10_list = [fm.test_metrics[epoch]['HR10'] for epoch in sorted(fm.test_metrics.keys())]
        tuning_timesteps.record(candidate, fold_hr10_list)
        print(f"[timesteps candidate {candidate}] Fold {tuning_fold}: HR@10 = {fold_hr10_list}")

    # Compute averages for each candidate.
    tuning_lr.average()
    tuning_optimizer.average()
    tuning_timesteps.average()

    # Print detailed average HR@10 per candidate.
    print("\nDetailed HR@10 (averaged over fold 1) per evaluation epoch:")
    print("Learning Rate Candidates:")
    for candidate in lr_candidates:
        avg_list = np.mean(np.array(tuning_lr.eval_dict[candidate]), axis=0)
        print(f"  Candidate {candidate}: {avg_list.tolist()}")

    print("Optimizer Candidates:")
    for candidate in optimizer_candidates:
        avg_list = np.mean(np.array(tuning_optimizer.eval_dict[candidate]), axis=0)
        print(f"  Candidate {candidate}: {avg_list.tolist()}")

    print("Timesteps Candidates:")
    for candidate in timesteps_candidates:
        avg_list = np.mean(np.array(tuning_timesteps.eval_dict[candidate]), axis=0)
        print(f"  Candidate {candidate}: {avg_list.tolist()}")

    # Determine best candidates (using the average at epoch 100).
    best_lr = tuning_lr.find_best()
    best_optimizer = tuning_optimizer.find_best()
    best_timesteps = tuning_timesteps.find_best()

    print("\nTuning complete.")
    print("Best learning rate:", best_lr)
    print("Best optimizer:", best_optimizer)
    print("Best timesteps:", best_timesteps)

    # Create and save a tuning summary.
    tuning_summary = TuningSummary()
    tuning_summary.add_metric(tuning_lr)
    tuning_summary.add_metric(tuning_optimizer)
    tuning_summary.add_metric(tuning_timesteps)

    print("\nTuning Summary:")
    print(tuning_summary)
    os.makedirs("./tune", exist_ok=True)
    with open("./tune/tuning_summary.txt", "w") as f:
        f.write(str(tuning_summary))
    print("Tuning summary saved in ./tune/tuning_summary.txt")

if __name__ == '__main__':
    main()

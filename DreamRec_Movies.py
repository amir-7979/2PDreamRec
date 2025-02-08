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
import logging

# Set logging level
logging.getLogger().setLevel(logging.INFO)

# =============================================================================
# 1. IMPORT MODEL & UTILITY MODULES
# =============================================================================
from Modules_ori import MovieTenc, MovieDiffusion, Tenc, diffusion, load_genres_predictor
from utility import extract_axis_1, calculate_hit

# =============================================================================
# 2. GLOBAL CONSTANTS AND DIRECTORY SETUP
# =============================================================================
MERGED_DATA_DIR = "data"  # directory containing train_fold*.df, val_fold*.df, test_fold*.df


# =============================================================================
# 3. DATASET DEFINITION
# =============================================================================
class MovieDataset(Dataset):
    """
    A PyTorch dataset for merged movie data.
    Expected CSV columns:
      - movie_seq: string representing a Python list of movie IDs
      - movie_len: integer sequence length (if missing, computed)
      - movie_target: target movie ID
      - genre_seq: string representing a list of genre IDs for the movies in the sequence
      - genre_target: target genre ID
    """

    def __init__(self, dataframe):
        self.movie_seq = dataframe['movie_seq'].tolist()
        if 'movie_len' in dataframe.columns:
            self.movie_len = dataframe['movie_len'].tolist()
        else:
            self.movie_len = [len(eval(s)) for s in self.movie_seq]
        if 'movie_target' in dataframe.columns:
            self.targets = dataframe['movie_target'].tolist()
        else:
            self.targets = dataframe['next'].tolist()
        # Genre branch (if available)
        self.genre_seq = dataframe['genre_seq'].tolist() if 'genre_seq' in dataframe.columns else None
        self.genre_targets = dataframe['genre_target'].tolist() if 'genre_target' in dataframe.columns else None

    def __len__(self):
        return len(self.movie_seq)

    def __getitem__(self, idx):
        seq = torch.tensor(eval(self.movie_seq[idx]), dtype=torch.long)
        length = torch.tensor(self.movie_len[idx], dtype=torch.long)
        target = torch.tensor(self.targets[idx], dtype=torch.long)
        if self.genre_seq is not None and self.genre_targets is not None:
            genre_seq = torch.tensor(eval(self.genre_seq[idx]), dtype=torch.long)
            genre_target = torch.tensor(self.genre_targets[idx], dtype=torch.long)
            return seq, length, target, genre_seq, genre_target
        else:
            return seq, length, target


# =============================================================================
# 4. HELPER FUNCTIONS: PAD & COLLATED BATCHES, VOCAB SIZES
# =============================================================================
def pad_or_truncate(seq, desired_length):
    if len(seq) > desired_length:
        return seq[-desired_length:]
    elif len(seq) < desired_length:
        return [0] * (desired_length - len(seq)) + seq
    else:
        return seq


def collate_fn(batch):
    seq_size = 10  # set your desired fixed sequence length
    movie_seqs, len_seqs, movie_targets, genre_seqs, genre_targets = zip(*batch)
    movie_seqs = [pad_or_truncate(seq.tolist(), seq_size) for seq in movie_seqs]
    len_seqs = [seq_size] * len(movie_seqs)
    genre_seqs = [pad_or_truncate(seq.tolist(), seq_size) for seq in genre_seqs]
    movie_seqs = torch.tensor(movie_seqs, dtype=torch.long)
    len_seqs = torch.tensor(len_seqs, dtype=torch.long)
    movie_targets = torch.tensor(movie_targets, dtype=torch.long)
    genre_seqs = torch.tensor(genre_seqs, dtype=torch.long)
    genre_targets = torch.tensor(genre_targets, dtype=torch.long)
    return movie_seqs, len_seqs, movie_targets, genre_seqs, genre_targets


def compute_vocab_sizes(df):
    movie_set = set()
    for seq_str in df["movie_seq"]:
        try:
            seq = eval(seq_str)
            movie_set.update(seq)
        except Exception as e:
            print(f"Error parsing movie sequence {seq_str}: {e}")
    movie_set.update(df["movie_target"].tolist())
    genre_set = set()
    for genre_seq_str in df["genre_seq"]:
        try:
            seq = eval(genre_seq_str)
            genre_set.update(seq)
        except Exception as e:
            print(f"Error parsing genre sequence {genre_seq_str}: {e}")
    genre_set.update(df["genre_target"].tolist())
    return len(movie_set), len(genre_set)


# =============================================================================
# 5. LOSS FUNCTION FOR MOVIE MODEL (FocalLoss)
# =============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# =============================================================================
# 6. EVALUATION FUNCTION (with Genre Integration)
# =============================================================================
def evaluate(model, genre_model, genre_diff, split_csv, diff, device):
    # For reproducibility
    fixed_seed = 100
    torch.manual_seed(fixed_seed)
    np.random.seed(fixed_seed)
    random.seed(fixed_seed)

    eval_data = pd.read_csv(os.path.join(MERGED_DATA_DIR, split_csv))
    batch_size = args.batch_size
    topk = [5, 10]
    total_samples = 0
    hit_purchase = np.zeros(len(topk))
    ndcg_purchase = np.zeros(len(topk))
    losses = []

    # Convert string lists to lists of ints.
    movie_seq = eval_data['movie_seq'].apply(lambda x: list(map(int, eval(x)))).tolist()
    movie_len = eval_data['movie_len'].values if 'movie_len' in eval_data.columns else np.array(
        [len(eval(x)) for x in eval_data['movie_seq'].tolist()])
    movie_target = eval_data['movie_target'].values
    genre_seq = eval_data['genre_seq'].apply(lambda x: list(map(int, eval(x)))).tolist()
    genre_target = eval_data['genre_target'].values

    for i in range(0, len(movie_seq), batch_size):
        seq_batch = torch.LongTensor(movie_seq[i:i + batch_size]).to(device)
        len_seq_batch = torch.LongTensor(movie_len[i:i + batch_size]).to(device)
        target_batch = torch.LongTensor(movie_target[i:i + batch_size]).to(device)
        genre_seq_batch = torch.LongTensor(genre_seq[i:i + batch_size]).to(device)
        genre_target_batch = torch.LongTensor(genre_target[i:i + batch_size]).to(device)

        # Movie branch:
        x_start = model.cacu_x(target_batch)
        h = model.cacu_h(seq_batch, len_seq_batch, args.p)
        n = torch.randint(0, args.timesteps, (seq_batch.shape[0],), device=device).long()
        # Genre branch:
        n_g = torch.randint(0, genre_diff.timesteps, (seq_batch.shape[0],), device=device).long()
        genre_x_start = genre_model.cacu_x(genre_target_batch)
        genre_h = genre_model.cacu_h(genre_seq_batch, len_seq_batch, args.p)
        _, genre_predicted_x = genre_diff.p_losses(genre_model, genre_x_start, genre_h, n_g, loss_type='l2')

        # Movie diffusion loss conditioned on genre branch output:
        loss1, predicted_x = diff.p_losses(model, x_start, h, n, genres_embd=genre_predicted_x, loss_type='l2')
        predicted_items = model.decoder(predicted_x)
        focal_loss = FocalLoss(alpha=0.5, gamma=7)
        loss2 = focal_loss(predicted_items, target_batch)
        loss = loss2 / 4
        losses.append(loss.item())

        # Get recommendations.
        prediction = F.softmax(predicted_items, dim=-1)
        _, topK = prediction.topk(10, dim=1, largest=True, sorted=True)
        topK = topK.cpu().detach().numpy()
        calculate_hit(target_batch, topK, topk, hit_purchase, ndcg_purchase)
        total_samples += len(target_batch)

    avg_loss = np.mean(losses) if losses else 0.0
    hr_list = hit_purchase / total_samples
    ndcg_list = ndcg_purchase / total_samples

    # Also print metrics for reference.
    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format(
        'HR@' + str(topk[0]), 'NDCG@' + str(topk[0]),
        'HR@' + str(topk[1]), 'NDCG@' + str(topk[1])))
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(
        hr_list[0], ndcg_list[0], hr_list[1], ndcg_list[1]))
    print(f'Loss: {avg_loss:.4f}')

    # Return a dictionary with all metric values.
    return {'loss': avg_loss, 'HR5': hr_list[0], 'NDCG5': ndcg_list[0],
            'HR10': hr_list[1], 'NDCG10': ndcg_list[1]}


# =============================================================================
# 7. CLASSES FOR RECORDING METRICS (Fold and Average)
# =============================================================================
class FoldMetrics:
    def __init__(self, fold_number):
        self.fold_number = fold_number
        self.train_losses = []  # list of tuples: (epoch, loss)
        self.val_metrics = {}  # dict: epoch -> metrics dict
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


# =============================================================================
# 8. MODIFIED TUNING CLASSES (Recording lists per candidate)
# =============================================================================
class TuningMetric:
    """
    Records, for each candidate, the list of HR@10 values at each evaluation checkpoint
    (e.g. epochs 10,20,...,100) from each fold.
    """

    def __init__(self, name, values):
        self.name = name  # e.g., 'lr'
        self.values = values  # candidate values list
        # For each candidate, we use a dict: key = evaluation epoch, value = list of HR@10 values.
        self.eval_dict = defaultdict(lambda: defaultdict(list))
        self.best_value = None

    def record(self, candidate, fold_hr10_list):
        for i, hr in enumerate(fold_hr10_list):
            self.eval_dict[candidate][(i + 1) * 10].append(hr)

    def average(self):
        self.avg_dict = {}
        for candidate in self.values:
            avg_list = []
            for epoch in sorted(self.eval_dict[candidate].keys()):
                avg_list.append(np.mean(self.eval_dict[candidate][epoch]))
            self.avg_dict[candidate] = avg_list
        return self.avg_dict

    def find_best(self):
        self.average()
        best = -np.inf
        for candidate in self.values:
            if 100 in self.eval_dict[candidate] and len(self.eval_dict[candidate][100]) > 0:
                avg_final = np.mean(self.eval_dict[candidate][100])
                if avg_final > best:
                    best = avg_final
                    self.best_value = candidate
        return self.best_value

    def __str__(self):
        s = f"TuningMetric for {self.name}:\n"
        for candidate in self.values:
            if candidate in self.eval_dict and self.eval_dict[candidate]:
                checkpoints = sorted(self.eval_dict[candidate].keys())
                avg_list = [np.mean(self.eval_dict[candidate][cp]) for cp in checkpoints]
                s += f"Candidate: {candidate}, HR@10 over eval epochs: {avg_list}\n"
            else:
                s += f"Candidate: {candidate}, No evaluations recorded\n"
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


# =============================================================================
# 9. TRAINING FUNCTION FOR ONE FOLD (with Genre Integration)
# =============================================================================
def train_fold(fold):
    print(f"\n========== Fold {fold} ==========")
    fold_metrics = FoldMetrics(fold)
    train_csv = f"train_fold{fold}.df"
    val_csv = f"val_fold{fold}.df"
    test_csv = f"test_fold{fold}.df"

    train_df = pd.read_csv(os.path.join(MERGED_DATA_DIR, train_csv))
    val_df = pd.read_csv(os.path.join(MERGED_DATA_DIR, val_csv))
    test_df = pd.read_csv(os.path.join(MERGED_DATA_DIR, test_csv))

    # Compute dynamic vocabulary sizes from the training set.
    movie_vocab_size_dynamic, genre_vocab_size_dynamic = compute_vocab_sizes(train_df)

    train_dataset = MovieDataset(train_df)
    val_dataset = MovieDataset(val_df)
    test_dataset = MovieDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    seq_size = 10  # fixed sequence length

    # Initialize the movie branch model.
    model = MovieTenc(args.hidden_factor, 4000, seq_size, args.dropout_rate, args.diffuser_type, device)
    diff = MovieDiffusion(args.timesteps, args.beta_start, args.beta_end, args.w)

    # Initialize the genre branch.
    genre_model = Tenc(args.hidden_factor, genre_vocab_size_dynamic, seq_size, args.dropout_rate, args.diffuser_type,
                       device)
    genre_diff = diffusion(100, args.beta_start, args.beta_end, args.w)
    genre_model, genre_diff = load_genres_predictor(genre_model)
    for param in genre_model.parameters():
        param.requires_grad = False  # freeze genre branch

    model.to(device)
    genre_model.to(device)

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

    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=2)

    # Training loop.
    for epoch in range(args.epoch):
        start_time = Time.time()
        model.train()
        for batch_data in train_loader:
            seq_batch, len_seq_batch, target_batch, genre_seq_batch, genre_target_batch = batch_data
            seq_batch = seq_batch.to(device)
            len_seq_batch = len_seq_batch.to(device)
            target_batch = target_batch.to(device)
            genre_seq_batch = genre_seq_batch.to(device)
            genre_target_batch = genre_target_batch.to(device)

            optimizer.zero_grad()
            # Movie branch forward.
            x_start = model.cacu_x(target_batch)
            h = model.cacu_h(seq_batch, len_seq_batch, args.p)
            n = torch.randint(0, args.timesteps, (seq_batch.shape[0],), device=device).long()
            # Genre branch forward.
            n_g = torch.randint(0, genre_diff.timesteps, (seq_batch.shape[0],), device=device).long()
            genre_x_start = genre_model.cacu_x(genre_target_batch)
            genre_h = genre_model.cacu_h(genre_seq_batch, len_seq_batch, args.p)
            _, genre_predicted_x = genre_diff.p_losses(genre_model, genre_x_start, genre_h, n_g, loss_type='l2')
            # Diffusion loss (movie branch conditioned on genre branch output)
            loss1, predicted_x = diff.p_losses(model, x_start, h, n, genres_embd=genre_predicted_x, loss_type='l2')
            predicted_items = model.decoder(predicted_x)
            focal_loss = FocalLoss(alpha=0.5, gamma=7)
            loss2 = focal_loss(predicted_items, target_batch)
            loss = loss2 / 4
            loss.backward()
            optimizer.step()
        fold_metrics.add_train_loss(epoch + 1, loss.item())
        if args.report_epoch:
            print(f"Fold {fold} Epoch {epoch + 1:03d}; Train loss: {loss.item():.4f}; "
                  f"Time: {Time.strftime('%H:%M:%S', Time.gmtime(Time.time() - start_time))}")
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                print(f"Fold {fold}: Validation Phase (Epoch {epoch + 1})")
                val_dict = evaluate(model, genre_model, genre_diff, val_csv, diff, device)
            fold_metrics.add_val_metrics(epoch + 1, val_dict)
            scheduler.step()

    with torch.no_grad():
        print(f"Fold {fold}: Test Phase")
        test_dict = evaluate(model, genre_model, genre_diff, test_csv, diff, device)
        print(f"Fold {fold}: Test Loss: {test_dict['loss']:.4f}, HR@5: {test_dict['HR5']:.4f}")
    fold_metrics.add_test_metrics(args.epoch, test_dict)

    os.makedirs("./models", exist_ok=True)
    torch.save(model.state_dict(), f"./models/movie_tenc_fold{fold}.pth")
    torch.save(diff, f"./models/movie_diff_fold{fold}.pth")

    # For tuning we return the entire fold_metrics so that we can extract HR@10 at each eval checkpoint.
    return fold_metrics


# =============================================================================
# 10. MAIN FUNCTION WITH IF-ELSE (Tuning vs. Normal 10-Fold CV)
# =============================================================================
def main():
    NUM_FOLDS = 10
    if args.tune:
        # Use only fold 1 for tuning.
        tuning_fold = 1
        # Define candidate values.
        lr_candidates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
        optimizer_candidates = ['adam', 'adamw', 'adagrad', 'rmsprop']
        timesteps_candidates = [i * 50 for i in range(1, 11)]

        # Create tuning metric objects.
        tuning_lr = TuningMetric("lr", lr_candidates)
        tuning_optimizer = TuningMetric("optimizer", optimizer_candidates)
        tuning_timesteps = TuningMetric("timesteps", timesteps_candidates)

        # --- Tuning learning rate using only fold 1 ---
        for candidate in tqdm(lr_candidates, desc="Tuning lr"):
            args.lr = candidate
            fm = train_fold(tuning_fold)
            # Extract HR@10 values from each evaluation checkpoint.
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
            if candidate in tuning_lr.eval_dict and tuning_lr.eval_dict[candidate]:
                checkpoints = sorted(tuning_lr.eval_dict[candidate].keys())
                avg_list = [np.mean(tuning_lr.eval_dict[candidate][cp]) for cp in checkpoints]
                print(f"  Candidate {candidate}: {avg_list}")
            else:
                print(f"  Candidate {candidate}: No evaluation recorded")
        print("Optimizer Candidates:")
        for candidate in optimizer_candidates:
            if candidate in tuning_optimizer.eval_dict and tuning_optimizer.eval_dict[candidate]:
                checkpoints = sorted(tuning_optimizer.eval_dict[candidate].keys())
                avg_list = [np.mean(tuning_optimizer.eval_dict[candidate][cp]) for cp in checkpoints]
                print(f"  Candidate {candidate}: {avg_list}")
            else:
                print(f"  Candidate {candidate}: No evaluation recorded")
        print("Timesteps Candidates:")
        for candidate in timesteps_candidates:
            if candidate in tuning_timesteps.eval_dict and tuning_timesteps.eval_dict[candidate]:
                checkpoints = sorted(tuning_timesteps.eval_dict[candidate].keys())
                avg_list = [np.mean(tuning_timesteps.eval_dict[candidate][cp]) for cp in checkpoints]
                print(f"  Candidate {candidate}: {avg_list}")
            else:
                print(f"  Candidate {candidate}: No evaluation recorded")

        # Determine best candidates (using the average HR@10 at epoch 100).
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

    else:
        # Full 10-Fold CV mode.
        fold_val_metrics = []
        fold_test_metrics = []
        for fold in range(1, NUM_FOLDS + 1):
            hr_val, hr_test = train_fold(fold)
            fold_val_metrics.append(hr_val)
            fold_test_metrics.append(hr_test)
            print("results:")
            print(f"Fold {fold}: Val HR@5 = {hr_val:.4f}, Test HR@5 = {hr_test:.4f}")
        print("\n========== 10-Fold Cross Validation Complete ==========")
        print("Average Val HR@5 across folds: {:.4f} ± {:.4f}".format(
            np.mean(fold_val_metrics), np.std(fold_val_metrics)))
        print("Average Test HR@5 across folds: {:.4f} ± {:.4f}".format(
            np.mean(fold_test_metrics), np.std(fold_test_metrics)))
        filename = "movie_not_une.txt"
        with open(filename, "w") as f:
            for fold in range(1, NUM_FOLDS + 1):
                f.write(
                    f"Fold {fold}: Val HR@5 = {fold_val_metrics[fold - 1]:.4f}, Test HR@5 = {fold_test_metrics[fold - 1]:.4f}\n")
        print(f"Results saved in {filename}")


if __name__ == '__main__':
    main()

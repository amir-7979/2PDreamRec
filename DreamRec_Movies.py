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
# IMPORTANT: Ensure that your Modules_ori.py defines the following classes:
#   - MovieTenc (a movie recommendation model that uses genre embeddings too)
#   - MovieDiffusion (the diffusion module for MovieTenc)
#   - Tenc (a vanilla transformer-based encoder used for the genre branch)
#   - diffusion (the base diffusion class)
#   - load_genres_predictor (loads pretrained genre model weights)
#
# Also, your utility module must provide functions like:
#   - extract_axis_1, calculate_hit
#
# (Adjust the import paths as needed.)
from Modules_ori import MovieTenc, MovieDiffusion, Tenc, diffusion, load_genres_predictor
from utility import extract_axis_1, calculate_hit

# =============================================================================
# 2. GLOBAL CONSTANTS AND DIRECTORY SETUP
# =============================================================================
# Directory where your merged CSV files (for each fold) reside.
MERGED_DATA_DIR = "data"  # e.g. the folder containing train_fold*.df, val_fold*.df, test_fold*.df


# =============================================================================
# 3. DATASET DEFINITION
# =============================================================================
class MovieDataset(Dataset):
    """
    A PyTorch dataset for merged movie data.
    Expected CSV columns (saved via your nested-fold merging script):
      - movie_seq (a string representing a Python list of movie IDs)
      - movie_len (an integer; training sequence length, e.g. 10)
      - movie_target (target movie ID)
      - genre_seq (a string representing a Python list of genre IDs for the movies in the seq)
      - genre_target (target genre ID)
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
        # Also load genre information if available:
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
# 4. LOSS FUNCTIONS AND METRIC HELPER
# =============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # inputs: [batch_size, num_classes]; targets: [batch_size]
        if inputs.ndim == 2 and targets.ndim == 1:
            ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        else:
            raise ValueError("Inputs should be [batch_size, num_classes] and targets [batch_size]")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class Metric:
    def __init__(self, name, values):
        self.name = name
        self.values = values
        self.eval_dict = defaultdict(list)
        self.bestOne = None

    def find_max_one(self):
        best = -np.inf
        for key, vals in self.eval_dict.items():
            temp = max(vals)
            if temp > best:
                best = temp
                self.bestOne = key


# =============================================================================
# 5. ARGUMENT PARSER & SEED SETUP
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Movie Diffusion Model with Genre Integration using 10-Fold CV"
    )
    parser.add_argument('--tune', action='store_true', default=False, help='Enable hyperparameter tuning.')
    parser.add_argument('--epoch', type=int, default=100, help='Number of epochs per fold.')
    parser.add_argument('--random_seed', type=int, default=100, help='Random seed.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64, help='Hidden/embedding size.')
    parser.add_argument('--timesteps', type=int, default=100, help='Diffusion timesteps.')
    parser.add_argument('--beta_end', type=float, default=0.02, help='Beta end for diffusion.')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='Beta start for diffusion.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--l2_decay', type=float, default=0.3, help='L2 regularization weight.')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device id.')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate.')
    parser.add_argument('--w', type=float, default=2.0, help='Weight for x_start update inside sampler.')
    parser.add_argument('--p', type=float, default=0.1, help='Probability used in cacu_h for random dropout.')
    parser.add_argument('--report_epoch', type=bool, default=True, help='Report metrics each epoch.')
    parser.add_argument('--diffuser_type', type=str, default='mlp1', help='Diffuser network type: mlp1 or mlp2.')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer type.')
    parser.add_argument('--beta_sche', nargs='?', default='exp', help='Beta schedule type.')
    parser.add_argument('--descri', type=str, default='', help='Run description.')
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


def pad_or_truncate(seq, desired_length):
    """
    Pads or truncates a list `seq` to the desired_length.
    If seq is shorter, pads with 0 (or another pad token).
    If seq is longer, truncates from the beginning.
    """
    if len(seq) > desired_length:
        return seq[-desired_length:]  # keep last desired_length items
    elif len(seq) < desired_length:
        return [0] * (desired_length - len(seq)) + seq  # pad at the beginning with 0s
    else:
        return seq


def collate_fn(batch):
    """
    Custom collate function to ensure each sequence is exactly seq_size long.
    Each item in the batch is assumed to be a tuple:
      (movie_seq, len_seq, movie_target, genre_seq, genre_target)
    """
    seq_size = 10  # This should match the seq_size you use for model initialization.

    movie_seqs, len_seqs, movie_targets, genre_seqs, genre_targets = zip(*batch)

    # Convert each movie_seq to a padded/truncated list of length seq_size.
    movie_seqs = [pad_or_truncate(seq.tolist(), seq_size) for seq in movie_seqs]
    # For lengths, simply use seq_size (if you intend for every sequence to be exactly that length)
    len_seqs = [seq_size] * len(movie_seqs)

    # Do the same for genre sequences.
    genre_seqs = [pad_or_truncate(seq.tolist(), seq_size) for seq in genre_seqs]

    # Convert everything back to tensors.
    movie_seqs = torch.tensor(movie_seqs, dtype=torch.long)
    len_seqs = torch.tensor(len_seqs, dtype=torch.long)
    movie_targets = torch.tensor(movie_targets, dtype=torch.long)
    genre_seqs = torch.tensor(genre_seqs, dtype=torch.long)
    genre_targets = torch.tensor(genre_targets, dtype=torch.long)

    return movie_seqs, len_seqs, movie_targets, genre_seqs, genre_targets


# =============================================================================
# 6. COMPUTE VOCAB SIZES
# =============================================================================
def compute_vocab_sizes(df):
    """
    Compute vocabulary sizes from the training DataFrame.

    Assumes df has the following columns:
      - 'movie_seq': string representation of a list of movie IDs
      - 'movie_target': target movie ID
      - 'genre_seq': string representation of a list of genre IDs
      - 'genre_target': target genre ID
    """
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
# 7. EVALUATION FUNCTION (with Genre Integration)
# =============================================================================
def evaluate(model, genre_model, genre_diff, split_csv, diff, device):
    eval_data = pd.read_csv(os.path.join(MERGED_DATA_DIR, split_csv))
    batch_size = args.batch_size
    topk = [5, 10]
    total_samples = 0
    hit_purchase = np.zeros(len(topk))
    ndcg_purchase = np.zeros(len(topk))
    losses = []
    # Convert string lists into Python lists of ints:
    movie_seq = eval_data['movie_seq'].apply(lambda x: list(map(int, eval(x)))).tolist()
    movie_len = (eval_data['movie_len'].values
                 if 'movie_len' in eval_data.columns
                 else np.array([len(eval(x)) for x in eval_data['movie_seq'].tolist()]))
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
        # Genre branch: use genre_diff.timesteps rather than a hardcoded 600.
        n_g = torch.randint(0, genre_diff.timesteps, (seq_batch.shape[0],), device=device).long()
        genre_x_start = genre_model.cacu_x(genre_target_batch)
        genre_h = genre_model.cacu_h(genre_seq_batch, len_seq_batch, args.p)
        _, genre_predicted_x = genre_diff.p_losses(genre_model, genre_x_start, genre_h, n_g, loss_type='l2')
        # Diffusion loss using the genre embedding as conditioning:
        loss1, predicted_x = diff.p_losses(model, x_start, h, n, genres_embd=genre_predicted_x, loss_type='l2')
        predicted_items = model.decoder(predicted_x)
        focal_loss = FocalLoss(alpha=0.5, gamma=7)

        loss2 = focal_loss(predicted_items, target_batch)
        loss = loss2/4
        losses.append(loss.item())
        # Get top-K recommendations:
        prediction = F.softmax(predicted_items, dim=-1)
        _, topK = prediction.topk(10, dim=1, largest=True, sorted=True)
        topK = topK.cpu().detach().numpy()
        calculate_hit(target_batch, topK, topk, hit_purchase, ndcg_purchase)
        total_samples += len(target_batch)

    avg_loss = sum(losses) / len(losses) if losses else 0.0
    hr_list = hit_purchase / total_samples
    ndcg_list = ndcg_purchase / total_samples
    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format(
        'HR@' + str(topk[0]), 'NDCG@' + str(topk[0]),
        'HR@' + str(topk[1]), 'NDCG@' + str(topk[1])))
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(
        hr_list[0], ndcg_list[0], hr_list[1], ndcg_list[1]))
    print(f'Loss: {avg_loss:.4f}')
    return avg_loss, hr_list[0]


# =============================================================================
# 8. TRAINING FUNCTION FOR ONE FOLD (with Genre Integration)
# =============================================================================
def train_fold(fold):
    print(f"\n========== Fold {fold} ==========")
    # File names for this fold (assumes merged CSV files are named accordingly)
    train_csv = f"train_fold{fold}.df"
    val_csv = f"val_fold{fold}.df"
    test_csv = f"test_fold{fold}.df"
    
    train_df = pd.read_csv(os.path.join(MERGED_DATA_DIR, train_csv))
    val_df = pd.read_csv(os.path.join(MERGED_DATA_DIR, val_csv))
    test_df = pd.read_csv(os.path.join(MERGED_DATA_DIR, test_csv))
    
    movie_vocab_size_dynamic, genre_vocab_size_dynamic = compute_vocab_sizes(train_df)


    train_dataset = MovieDataset(train_df)
    val_dataset = MovieDataset(val_df)
    test_dataset = MovieDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Set the maximum sequence length (e.g. 10)
    seq_size = 10

    # Initialize the movie model and its diffusion module.
    # For the movie branch, pass the dynamically computed movie vocabulary size.
    model = MovieTenc(args.hidden_factor, 4000, seq_size, args.dropout_rate, args.diffuser_type,
                      device)
    diff = MovieDiffusion(args.timesteps, args.beta_start, args.beta_end, args.w)

    # Initialize the genre branch.
    # Here Tenc is used to encode genre sequences and diffusion to condition on genre.
    genre_model = Tenc(args.hidden_factor, genre_vocab_size_dynamic, seq_size, args.dropout_rate, args.diffuser_type,
                       device)
    genre_diff = diffusion(100, args.beta_start, args.beta_end, args.w)
    genre_model, genre_diff = load_genres_predictor(genre_model)
    # Freeze genre model parameters.
    for param in genre_model.parameters():
        param.requires_grad = False

    model.to(device)
    genre_model.to(device)

    # Choose optimizer based on args.optimizer
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

    for epoch in range(args.epoch):
        start_time = Time.time()
        model.train()
        for batch_data in train_loader:
            # Expect each batch to yield: (movie_seq, movie_len, movie_target, genre_seq, genre_target)
            seq_batch, len_seq_batch, target_batch, genre_seq_batch, genre_target_batch = batch_data
            seq_batch = seq_batch.to(device)
            len_seq_batch = len_seq_batch.to(device)
            target_batch = target_batch.to(device)
            genre_seq_batch = genre_seq_batch.to(device)
            genre_target_batch = genre_target_batch.to(device)

            optimizer.zero_grad()
            # Movie branch forward:
            x_start = model.cacu_x(target_batch)
          


            h = model.cacu_h(seq_batch, len_seq_batch, args.p)
            n = torch.randint(0, args.timesteps, (seq_batch.shape[0],), device=device).long()
            # Genre branch forward:
            n_g = torch.randint(0, 100, (seq_batch.shape[0],), device=device).long()
            genre_x_start = genre_model.cacu_x(genre_target_batch)
            genre_h = genre_model.cacu_h(genre_seq_batch, len_seq_batch, args.p)
            _, genre_predicted_x = genre_diff.p_losses(genre_model, genre_x_start, genre_h, n_g, loss_type='l2')
            # Diffusion loss (movie branch conditioned on genre branch output)
            loss1, predicted_x = diff.p_losses(model, x_start, h, n, genres_embd=genre_predicted_x, loss_type='l2')
            predicted_items = model.decoder(predicted_x)
            focal_loss = FocalLoss(alpha=0.5, gamma=7)
            loss2 = focal_loss(predicted_items, target_batch)
            loss = loss2/4

            loss.backward()
            optimizer.step()
        if args.report_epoch:
            print(f"Fold {fold} Epoch {epoch:03d}; Train loss: {loss.item():.4f}; "
                  f"Time: {Time.strftime('%H:%M:%S', Time.gmtime(Time.time() - start_time))}")

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                print(f"Fold {fold}: Validation Phase")
                avg_loss, hr_val = evaluate(model, genre_model, genre_diff, val_csv, diff, device)
                print(f"Fold {fold}: Val Loss: {avg_loss:.4f}, HR@5: {hr_val:.4f}")
            scheduler.step()

    # After training, evaluate on the test set.
    with torch.no_grad():
        print(f"Fold {fold}: Test Phase")
        avg_loss_test, hr_test = evaluate(model, genre_model, genre_diff, test_csv, diff, device)
        print(f"Fold {fold}: Test Loss: {avg_loss_test:.4f}, HR@5: {hr_test:.4f}")

    os.makedirs("./models", exist_ok=True)
    torch.save(model.state_dict(), f"./models/movie_tenc_fold{fold}.pth")
    torch.save(diff, f"./models/movie_diff_fold{fold}.pth")

    return hr_val, hr_test


# =============================================================================
# 9. MAIN FUNCTION (10-Fold CV or Tuning Mode)
# =============================================================================
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
        torch.save(best_metrics, './tune/metrics_m.dict')
        print("Tuning complete. Best metrics:")
        for m in best_metrics:
            print(f"{m.name}: {m.bestOne}")
    else:
        args.lr = 0.001
        args.optimizer = 'adamw'
        metrics = [Metric(name='timesteps', values=[100])]
        best_metrics = []
        for fold in range(1, NUM_FOLDS + 1):
            hr_val, hr_test = train_fold(fold)
            fold_val_metrics.append(hr_val)
            fold_test_metrics.append(hr_test)
        print("\n========== 10-Fold Cross Validation Complete ==========")
        print("Average Val HR@5 across folds: {:.4f} ± {:.4f}".format(
            np.mean(fold_val_metrics), np.std(fold_val_metrics)))
        print("Average Test HR@5 across folds: {:.4f} ± {:.4f}".format(
            np.mean(fold_test_metrics), np.std(fold_test_metrics)))


if __name__ == '__main__':
    main()

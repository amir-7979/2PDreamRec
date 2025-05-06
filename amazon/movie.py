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
from adabelief_pytorch import AdaBelief
from torch_optimizer import Lamb, AdamP, RAdam, Shampoo, DiffGrad
from torch_optimizer import Ranger
import random
import logging
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from Modules_ori import MovieTenc, MovieDiffusion, Tenc, diffusion, load_genres_predictor, get_top_genres, filter_movie_scores
from utility import extract_axis_1, calculate_hit
from recorders import LossTuningRecorder, MetricsRecorder, LossRecorder, FoldMetrics, AverageMetrics

logging.getLogger().setLevel(logging.INFO)
MERGED_DATA_DIR = "data"
movie_genre_mapping = None
genre_movie_mapping = None

############################################
# Argument Parsing and Setup
############################################
def parse_args():
    parser = argparse.ArgumentParser(description="Run Movie Prediction with Diffusion + Focal Loss using 10-Fold CV on merged data.")
    parser.add_argument('--tune', action='store_true', default=False, help='Enable tuning.')
    parser.add_argument('--no-tune', action='store_false', dest='tune', help='Disable tuning.')
    parser.add_argument('--epoch', type=int, default=60, help='Number of epochs per fold.')
    parser.add_argument('--random_seed', type=int, default=100, help='Random seed.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64, help='Embedding/hidden size.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--optimizer', type=str, default='radam',help='Optimizer type: [adam, adamw, adagrad, rmsprop, lamb, adamp, radam, adabelief, nadam].')
    parser.add_argument('--timesteps', type=int, default=100, help='Timesteps for diffusion.')
    parser.add_argument('--dropout_rate', type=float, default=1e-8, help='Dropout rate.')
    parser.add_argument('--eps', type=float, default=1e-4, help='eps')
    parser.add_argument('--l2_decay', type=float, default=1e-2, help='L2 loss regularization coefficient.')
    parser.add_argument('--scheduler_factor', type=float, default=0.5, help='Factor for ReduceLROnPlateau scheduler.')
    parser.add_argument('--scheduler', type=str, default='step', help='Scheduler type: [reduce_on_plateau, cosine, step].')

    parser.add_argument('--beta_end', type=float, default=0.02, help='Beta end of diffusion.')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='Beta start of diffusion.')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device id.')
    parser.add_argument('--w', type=float, default=2.0, help='Weight used in x_start update inside sampler.')
    parser.add_argument('--p', type=float, default=0.2, help='Probability used in cacu_h for random dropout.')
    parser.add_argument('--report_epoch', type=bool, default=True, help='Whether to report metrics each epoch.')
    parser.add_argument('--diffuser_type', type=str, default='mlp1', help='Type of diffuser network: [mlp1, mlp2].')
    parser.add_argument('--beta_sche', nargs='?', default='linear', help='Beta schedule: [linear, exp, cosine, sqrt].')
    parser.add_argument('--descri', type=str, default='', help='Description of the run.')
    parser.add_argument('--exp_length', type=int, default=None, help='Training sequence length (without target) for experimental runs.')
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

############################################
# Dataset Definition for Movie Model
############################################

class MovieDataset(Dataset):
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
        self.genre_seq = dataframe['genre_seq'].tolist() if 'genre_seq' in dataframe.columns else None
        self.genre_targets = dataframe['genre_target'].tolist() if 'genre_target' in dataframe.columns else None

    def __len__(self):
        return len(self.movie_seq)

    def __getitem__(self, idx):
        seq_data = self.movie_seq[idx]
        if isinstance(seq_data, str):
            seq_data = eval(seq_data)
        seq = torch.tensor(seq_data, dtype=torch.long)

        
        target = torch.tensor(self.targets[idx], dtype=torch.long)
        seq = seq[seq != target]
        
        length = torch.tensor(len(seq), dtype=torch.long)
        if self.genre_seq is not None and self.genre_targets is not None:
            genre_data = self.genre_seq[idx]
            if isinstance(genre_data, str):
                genre_data = eval(genre_data)
            genre_seq = torch.tensor(genre_data, dtype=torch.long)

            genre_target = torch.tensor(self.genre_targets[idx], dtype=torch.long)
            return seq, length, target, genre_seq, genre_target
        else:
            return seq, length, target

def movie_collate_fn(batch, fixed_length):
    seqs, lengths, targets, genre_seqs, genre_targets = zip(*batch)
    padded_seqs = []
    for i, s in enumerate(seqs):
        if s.numel() == 0:  # if the sequence is empty
            padded = torch.zeros(fixed_length, dtype=s.dtype)
        elif s.size(0) < fixed_length:
            pad_size = fixed_length - s.size(0)
            padded = torch.cat([s, torch.full((pad_size,), s[-1], dtype=s.dtype)])
        else:
            padded = s[:fixed_length]
        if padded.size(0) != fixed_length:
            print(f"Warning: Sample {i} padded sequence length is {padded.size(0)} (expected {fixed_length}). Original length was {s.size(0)}.")
        padded_seqs.append(padded)
    padded_seqs = torch.stack(padded_seqs)
    new_lengths = torch.tensor([min(l.item(), fixed_length) for l in lengths])
    targets = torch.stack(targets)
    if genre_seqs[0] is not None:
        padded_genre_seqs = []
        for i, s in enumerate(genre_seqs):
            if s.numel() == 0:
                padded = torch.zeros(fixed_length, dtype=s.dtype)
            elif s.size(0) < fixed_length:
                pad_size = fixed_length - s.size(0)
                padded = torch.cat([s, torch.full((pad_size,), s[-1], dtype=s.dtype)])
            else:
                padded = s[:fixed_length]
            if padded.size(0) != fixed_length:
                print(f"Warning: Sample {i} padded genre sequence length is {padded.size(0)} (expected {fixed_length}). Original length was {s.size(0)}.")
            padded_genre_seqs.append(padded)
        padded_genre_seqs = torch.stack(padded_genre_seqs)
        genre_targets = torch.stack(genre_targets)
    else:
        padded_genre_seqs = None
        genre_targets = None

    return padded_seqs, new_lengths, targets, padded_genre_seqs, genre_targets

############################################
# Loss Function: FocalLoss
############################################

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

############################################
# Evaluation Function (with Genre Integration) for Movie Model
############################################

def evaluate(model, genre_model, genre_diff, split_csv, diff, device):
    eval_data = pd.read_pickle(os.path.join(MERGED_DATA_DIR, split_csv))
    batch_size = args.batch_size
    topk = [5, 10, 20]
    total_samples = 0
    hit_purchase = np.zeros(len(topk))
    ndcg_purchase = np.zeros(len(topk))
    losses = []
    movie_seq = eval_data['movie_seq'].tolist()

    movie_len = (eval_data['movie_len'].values if 'movie_len' in eval_data.columns
                 else np.array([len(eval(x)) for x in eval_data['movie_seq'].tolist()]))
    movie_target = eval_data['movie_target'].values
    genre_seq = eval_data['genre_seq'].tolist()
    genre_target = eval_data['genre_target'].values
    for i in range(0, len(movie_seq), batch_size):
        seq_batch = torch.LongTensor(movie_seq[i:i + batch_size]).to(device)
        len_seq_batch = torch.LongTensor(movie_len[i:i + batch_size]).to(device)
        target_batch = torch.LongTensor(movie_target[i:i + batch_size]).to(device)
        genre_seq_batch = torch.LongTensor(genre_seq[i:i + batch_size]).to(device)
        genre_target_batch = torch.LongTensor(genre_target[i:i + batch_size]).to(device)
        x_start = model.cacu_x(target_batch)
        h = model.cacu_h(seq_batch, len_seq_batch, args.p)
        n = torch.randint(0, args.timesteps, (seq_batch.shape[0],), device=device).long()
        n_g = torch.randint(0, genre_diff.timesteps, (seq_batch.shape[0],), device=device).long()
        genre_x_start = genre_model.cacu_x(genre_target_batch)
        genre_h = genre_model.cacu_h(genre_seq_batch, len_seq_batch, args.p)
        _, genre_predicted_x = genre_diff.p_losses(genre_model, genre_x_start, genre_h, n_g, loss_type='l2')
        loss1, predicted_x = diff.p_losses(model, x_start, h, n, genres_embd=genre_predicted_x, loss_type='l2')
        predicted_items = model.decoder(predicted_x)
        focal_loss = FocalLoss(alpha=0.08, gamma=10)
        loss2 = focal_loss(predicted_items, target_batch)
        loss = loss2
        losses.append(loss.item())
        prediction = torch.softmax(predicted_items, dim=-1)
        _, topK = prediction.topk(20, dim=1, largest=True, sorted=True)
        topK = topK.cpu().detach().numpy()
        calculate_hit(target_batch, topK, topk, hit_purchase, ndcg_purchase)
        total_samples += len(target_batch)
    avg_loss = np.mean(losses) if losses else 0.0
    hr_list = hit_purchase / total_samples
    ndcg_list = ndcg_purchase / total_samples
    print('{:<10s} {:<10s} {:<10s} {:<10s} {:<10s} {:<10s}'.format(
        'HR@' + str(topk[0]), 'NDCG@' + str(topk[0]),
        'HR@' + str(topk[1]), 'NDCG@' + str(topk[1]),
        'HR@' + str(topk[2]), 'NDCG@' + str(topk[2]),))
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(
        hr_list[0], ndcg_list[0], hr_list[1], ndcg_list[1], hr_list[2], ndcg_list[2]))
    print(f'Loss: {avg_loss:.4f}')
    return {'loss': avg_loss, 'HR5': hr_list[0], 'NDCG5': ndcg_list[0],
            'HR10': hr_list[1], 'NDCG10': ndcg_list[1], 'HR10': hr_list[2], 'NDCG10': ndcg_list[2]}

############################################
# Original Training Function for One Fold (Movie Model)
############################################

def train_fold(fold):
    global movie_genre_mapping, genre_movie_mapping
    print(f"\n========== Fold {fold} ==========")
    fold_metrics = FoldMetrics(fold)
    train_csv = f"train_fold{fold}.df"
    test_csv = f"test_fold{fold}.df"
    train_df = pd.read_pickle(os.path.join(MERGED_DATA_DIR, train_csv))
    test_df = pd.read_pickle(os.path.join(MERGED_DATA_DIR, test_csv))
    train_dataset = MovieDataset(train_df)
    test_dataset = MovieDataset(test_df)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: movie_collate_fn(batch, fixed_length=10))
    statics_file = os.path.join(MERGED_DATA_DIR, "statics.csv")
    if os.path.exists(statics_file):
        statics_df = pd.read_csv(statics_file)
        statics = dict(zip(statics_df["statistic"], statics_df["value"]))
        movie_vocab_size_dynamic = 275745
        print(movie_vocab_size_dynamic)
        genre_vocab_size_dynamic = int(statics["num_genres"])
        seq_size = int(statics.get("train_seq_length", 200))
    else:
        movie_vocab_size_dynamic = 275745
        genre_vocab_size_dynamic = 200
        seq_size = 10
    model = MovieTenc(args.hidden_factor, movie_vocab_size_dynamic, seq_size, args.dropout_rate, args.diffuser_type, device=device).to(device)
    diff = MovieDiffusion(args.timesteps, args.beta_start, args.beta_end, args.w)
    genre_model = Tenc(args.hidden_factor, genre_vocab_size_dynamic, seq_size, args.dropout_rate, args.diffuser_type, device)
    genre_diff = diffusion(100, args.beta_start, args.beta_end, args.w)
    genre_model, genre_diff = load_genres_predictor(genre_model)
    for param in genre_model.parameters():
        param.requires_grad = False
    model.to(device)
    genre_model.to(device)
    if args.optimizer == 'adamp':
        optimizer = AdamP(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.l2_decay)
    elif args.optimizer == 'radam':
        optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=args.l2_decay)
    elif args.optimizer == 'lamb':
        optimizer = Lamb(model.parameters(), lr=args.lr, eps=args.eps, weight_decay=args.l2_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, eps=args.eps, weight_decay=args.l2_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, eps=args.eps, weight_decay=args.l2_decay)
        
    if args.scheduler == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.scheduler_factor, eps=args.scheduler_eps)
    elif args.scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-5)
    elif args.scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=args.scheduler_factor)
    else:
        scheduler = None
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
            x_start = model.cacu_x(target_batch)
            h = model.cacu_h(seq_batch, len_seq_batch, args.p)
            n = torch.randint(0, args.timesteps, (seq_batch.shape[0],), device=device).long()
            n_g = torch.randint(0, genre_diff.timesteps, (seq_batch.shape[0],), device=device).long()
            genre_x_start = genre_model.cacu_x(genre_target_batch)
            genre_h = genre_model.cacu_h(genre_seq_batch, len_seq_batch, args.p)
            _, genre_predicted_x = genre_diff.p_losses(genre_model, genre_x_start, genre_h, n_g, loss_type='l2')
            loss1, predicted_x = diff.p_losses(model, x_start, h, n, genres_embd=genre_predicted_x, loss_type='l2')
            predicted_items = model.decoder(predicted_x)
            focal_loss = FocalLoss(alpha=0.08, gamma=10)
            loss2 = focal_loss(predicted_items, target_batch)
            loss = loss2
            loss.backward()
            optimizer.step()
        fold_metrics.add_train_loss(epoch + 1, loss.item())
        if args.report_epoch:
            print(f"Fold {fold} Epoch {epoch + 1:03d}; Train loss: {loss.item():.4f}; Time: {Time.strftime('%H:%M:%S', Time.gmtime(Time.time() - start_time))}")
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                print(f"Fold {fold}: Evaluation at Epoch {epoch + 1}")
                test_dict = evaluate(model, genre_model, genre_diff, test_csv, diff, device)
            fold_metrics.add_test_metrics(epoch + 1, test_dict)
            if args.scheduler == 'reduce_on_plateau':
                scheduler.step(test_dict['loss'])
            elif scheduler is not None:
                scheduler.step()
    with torch.no_grad():
        print(f"Fold {fold}: Final Test Phase")
        final_test = evaluate(model, genre_model, genre_diff, test_csv, diff, device)
        print(f"Fold {fold}: Final Test Loss: {final_test['loss']:.4f}, HR@5: {final_test['HR5']:.4f}")
    fold_metrics.add_test_metrics(args.epoch, final_test)
    os.makedirs("./models", exist_ok=True)
    torch.save(model.state_dict(), f"./models/movie_tenc_fold{fold}.pth")
    torch.save(diff, f"./models/movie_diff_fold{fold}.pth")
    return fold_metrics


############################################
# Main Function
############################################

def main():
    NUM_FOLDS = 1
    global movie_genre_mapping, genre_movie_mapping
    mapping_csv = os.path.join(MERGED_DATA_DIR, "movie_to_genre_mapping.csv")
    genre_mapping_json = os.path.join(MERGED_DATA_DIR, "genre_movie_mapping.json")
    if os.path.exists(mapping_csv):
        mapping_df = pd.read_csv(mapping_csv)
        statics_file = os.path.join(MERGED_DATA_DIR, "statics.csv")
        if os.path.exists(statics_file):
            statics_df = pd.read_csv(statics_file)
            statics = dict(zip(statics_df["statistic"], statics_df["value"]))
            used_movie_count = int(statics["num_movies"])
        else:
            used_movie_count = mapping_df.shape[0]
        filtered_mapping_df = mapping_df.head(used_movie_count)
        movie_genre_mapping = torch.tensor(filtered_mapping_df["genreId"].tolist(), device=device)
        print("Loaded and filtered movie-to-genre mapping with shape:", movie_genre_mapping.shape)
        print("Mapping tensor:", movie_genre_mapping)
    else:
        print("Mapping file not found. Proceeding without genre filtering.")
        movie_genre_mapping = None
        
    genre_mapping_json = os.path.join(MERGED_DATA_DIR, "genre_movie_mapping.json")
    if os.path.exists(genre_mapping_json):
        with open(genre_mapping_json, "r", encoding="utf-8") as f:
            genre_movie_mapping = json.load(f)
        print("Loaded genre-to-movies mapping")
    else:
        print("Genre mapping JSON file not found")
        genre_movie_mapping = None

    if args.tune:
        tuning_fold = 1
        lr_candidates = [0.05, 0.01, 0.005, 0.001]
        tuning_lr_recorder = LossTuningRecorder("lr", candidates=lr_candidates, save_dir="item")
        for candidate in tqdm(lr_candidates, desc="Tuning lr based on loss"):
            args.lr = candidate
            fm = train_fold(tuning_fold)
            for epoch, metrics in fm.test_metrics.items():
                tuning_lr_recorder.record(candidate, epoch, metrics['loss'])
            final_epoch = max(fm.test_metrics.keys())
            print(f"[lr candidate {candidate}] Fold {tuning_fold}: Loss = {fm.test_metrics[final_epoch]['loss']:.4f}")
        best_lr = tuning_lr_recorder.find_best_loss(eval_epoch=100)
        print("\nBest learning rate based on loss:", best_lr)
        args.lr = best_lr
        tuning_lr_recorder.save_to_file("./item/tuning_lr.json")

        optimizer_candidates = ['lamb', 'adamw', 'radam', 'adamp']
        tuning_optimizer_recorder = LossTuningRecorder("optimizer", candidates=optimizer_candidates, save_dir="item")
        for candidate in tqdm(optimizer_candidates, desc="Tuning optimizer based on loss"):
            args.optimizer = candidate
            fm = train_fold(tuning_fold)
            for epoch, metrics in fm.test_metrics.items():
                tuning_optimizer_recorder.record(candidate, epoch, metrics['loss'])
            final_epoch = max(fm.test_metrics.keys())
            print(f"[optimizer candidate {candidate}] Fold {tuning_fold}: Loss = {fm.test_metrics[final_epoch]['loss']:.4f}")
        best_optimizer = tuning_optimizer_recorder.find_best_loss(eval_epoch=100)
        print("\nBest optimizer based on loss:", best_optimizer)
        args.optimizer = best_optimizer
        tuning_optimizer_recorder.save_to_file("./item/tuning_optimizer.json")


        timesteps_candidates = [100, 150, 200, 250, 300]
        tuning_timesteps_recorder = LossTuningRecorder("timesteps", candidates=timesteps_candidates, save_dir="item")
        for candidate in tqdm(timesteps_candidates, desc="Tuning timesteps based on loss"):
            args.timesteps = candidate
            fm = train_fold(tuning_fold)
            for epoch, metrics in fm.test_metrics.items():
                tuning_timesteps_recorder.record(candidate, epoch, metrics['loss'])
            final_epoch = max(fm.test_metrics.keys())
            print(f"[timesteps candidate {candidate}] Fold {tuning_fold}: Loss = {fm.test_metrics[final_epoch]['loss']:.4f}")
        best_timesteps = tuning_timesteps_recorder.find_best_loss(eval_epoch=100)
        print("\nBest timesteps based on loss:", best_timesteps)
        args.timesteps = best_timesteps
        tuning_timesteps_recorder.save_to_file("./item/tuning_timesteps.json")


        dropout_candidates = [1e-1, 1e-2, 1e-4, 1e-8]
        tuning_dropout_recorder = LossTuningRecorder("dropout_rate", candidates=dropout_candidates, save_dir="item")
        for candidate in tqdm(dropout_candidates, desc="Tuning dropout_rate based on loss"):
            args.dropout_rate = candidate
            fm = train_fold(tuning_fold)
            for epoch, metrics in fm.test_metrics.items():
                tuning_dropout_recorder.record(candidate, epoch, metrics['loss'])
            final_epoch = max(fm.test_metrics.keys())
            print(f"[dropout_rate candidate {candidate}] Fold {tuning_fold}: Loss = {fm.test_metrics[final_epoch]['loss']:.4f}")
        best_dropout = tuning_dropout_recorder.find_best_loss(eval_epoch=100)
        print("\nBest dropout_rate based on loss:", best_dropout)
        args.dropout_rate = best_dropout
        tuning_dropout_recorder.save_to_file("./item/tuning_dropout.json")

        l2_candidates = [1e-1, 1e-2, 1e-4, 1e-8]
        tuning_l2_recorder = LossTuningRecorder("l2_decay", candidates=l2_candidates, save_dir="item")
        for candidate in tqdm(l2_candidates, desc="Tuning l2_decay based on loss"):
            args.l2_decay = candidate
            fm = train_fold(tuning_fold)
            for epoch, metrics in fm.test_metrics.items():
                tuning_l2_recorder.record(candidate, epoch, metrics['loss'])
            final_epoch = max(fm.test_metrics.keys())
            print(f"[l2_decay candidate {candidate}] Fold {tuning_fold}: Loss = {fm.test_metrics[final_epoch]['loss']:.4f}")
        best_l2 = tuning_l2_recorder.find_best_loss(eval_epoch=100)
        print("\nBest l2_decay based on loss:", best_l2)
        args.l2_decay = best_l2
        tuning_l2_recorder.save_to_file("./item/tuning_l2.json")

        eps_candidates = [1e-1, 1e-2, 1e-4, 1e-8]
        tuning_eps_recorder = LossTuningRecorder("eps", candidates=eps_candidates, save_dir="item")
        for candidate in tqdm(eps_candidates, desc="Tuning eps based on loss"):
            args.eps = candidate
            fm = train_fold(tuning_fold)
            for epoch, metrics in fm.test_metrics.items():
                tuning_eps_recorder.record(candidate, epoch, metrics['loss'])
            final_epoch = max(fm.test_metrics.keys())
            print(f"[eps candidate {candidate}] Fold {tuning_fold}: Loss = {fm.test_metrics[final_epoch]['loss']:.4f}")
        best_eps = tuning_eps_recorder.find_best_loss(eval_epoch=100)
        print("\nBest eps based on loss:", best_eps)
        args.eps = best_eps
        tuning_eps_recorder.save_to_file("./item/tuning_eps.json")



        best_candidates = {
            "lr": best_lr,
            "timesteps": best_timesteps,
            "optimizer": best_optimizer,
            "dropout_rate": best_dropout,
            "l2_decay": best_l2,
            "eps": best_eps
        }
        with open("./item/best_candidates.json", "w") as f:
            json.dump(best_candidates, f, indent=2)
        print("Best candidates saved to ./item/best_candidates.json")
    
    fold_metrics_list = []
    for fold in range(1, NUM_FOLDS + 1):
        fm = train_fold(fold)
        fold_metrics_list.append(fm)
        print("Results for Fold", fold)
        print(fm)
        os.makedirs("./item", exist_ok=True)
    avg_metrics = AverageMetrics()
    for fm in fold_metrics_list:
        avg_metrics.add_fold_metrics(fm)
    avg_metrics.compute_averages()
    print("\n========== Average Metrics Across Folds ==========")
    print(avg_metrics)
    with open("item/average_metrics.txt", "w") as f:
        f.write(str(avg_metrics))
    print("Full average metrics saved to item/average_metrics.txt")
    loss_recorder = LossRecorder(save_dir="item")
    for fm in fold_metrics_list:
        sorted_train = [loss for (ep, loss) in sorted(fm.train_losses, key=lambda x: x[0])]
        sorted_test = [fm.test_metrics[ep]['loss'] for ep in sorted(fm.test_metrics.keys())]
        loss_recorder.add_fold(fm.fold_number, sorted_train, sorted_test)
    loss_recorder.save_to_file("item/loss_data.json")
    avg_train, avg_test = loss_recorder.compute_average_losses()
    epochs_train = sorted(avg_train.keys())
    np.savetxt("item/avg_train_loss.txt", np.array([avg_train[e] for e in epochs_train]))
    print("Average training losses saved to item/avg_train_loss.txt")
    metrics_recorder = MetricsRecorder(save_dir="item")
    for fm in fold_metrics_list:
        metrics_recorder.add_fold(fm)
    metrics_recorder.save_to_file("item/average_test_metrics.txt")

       
if __name__ == '__main__':
    main()

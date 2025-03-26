import os
import time as Time
import argparse
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import json
import matplotlib.pyplot as plt
import logging
from Modules_ori import MovieDiffusion, Tenc, MovieTenc, load_genres_predictor, diffusion, Tenc
from utility import extract_axis_1, calculate_hit
from recorders import LossRecorder, MetricsRecorder, TuningRecorder, FoldMetrics, AverageMetrics
from adabelief_pytorch import AdaBelief
from torch_optimizer import Lamb
from torch.optim.lr_scheduler import ReduceLROnPlateau

logging.getLogger().setLevel(logging.INFO)
MERGED_DATA_DIR = "data"

############################################
# Dataset Definition for Genre Model
############################################
class GenreDataset(Dataset):
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
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--optimizer', type=str, default='nadam', help='Optimizer type: [adam, adamw, adagrad, rmsprop].')
    parser.add_argument('--timesteps', type=int, default=100, help='Timesteps for diffusion.')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate.')
    parser.add_argument('--eps', type=float, default=5e-5, help='L2 loss regularization coefficient.')
    parser.add_argument('--l2_decay', type=float, default=5e-5, help='L2 loss regularization coefficient.')
    parser.add_argument('--factor', type=float, default=0.5, help='L2 loss regularization coefficient.')

    parser.add_argument('--hidden_factor', type=int, default=64, help='Embedding/hidden size.')
    parser.add_argument('--beta_end', type=float, default=0.02, help='Beta end of diffusion.')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='Beta start of diffusion.')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device id.')
    parser.add_argument('--w', type=float, default=2.0, help='Weight used in x_start update inside sampler.')
    parser.add_argument('--p', type=float, default=0.1, help='Probability used in cacu_h for random dropout.')
    parser.add_argument('--report_epoch', type=bool, default=True, help='Whether to report metrics each epoch.')
    parser.add_argument('--diffuser_type', type=str, default='mlp1', help='Type of diffuser network: [mlp1, mlp2].')
    parser.add_argument('--beta_sche', nargs='?', default='linear', help='Beta schedule: [linear, exp, cosine, sqrt].')
    parser.add_argument('--descri', type=str, default='', help='Description of the run.')
    # New argument: exp_length to override sequence length for experimental folds.
    parser.add_argument('--exp_length', type=int, default=None, help='Sequence length for experimental runs (training sequence length, i.e. without target).')
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
# Evaluation Function for Genre Model
############################################
def evaluate(model, diff, dataset_split, device):
    eval_data = pd.read_csv(os.path.join(MERGED_DATA_DIR, dataset_split))
    batch_size = args.batch_size
    topk = [5, 10]
    total_samples = 0
    hit_purchase = np.zeros(len(topk))
    ndcg_purchase = np.zeros(len(topk))
    losses = []
    genre_seq = eval_data['genre_seq'].apply(lambda x: list(map(int, eval(x)))).tolist()
    genre_len = (eval_data['genre_len'].values if 'genre_len' in eval_data.columns
                 else np.array([len(eval(x)) for x in eval_data['genre_seq'].tolist()]))
    genre_target = eval_data['genre_target'].values
    for i in range(0, len(genre_seq), batch_size):
        seq_batch = torch.LongTensor(genre_seq[i:i + batch_size]).to(device)
        len_seq_batch = torch.LongTensor(genre_len[i:i + batch_size]).to(device)
        target_batch = torch.LongTensor(genre_target[i:i + batch_size]).to(device)
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
    print('{:<10s} {:<10s} {:<10s} {:<10s}'.format(
        'HR@' + str(topk[0]), 'NDCG@' + str(topk[0]),
        'HR@' + str(topk[1]), 'NDCG@' + str(topk[1])))
    print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(
        hr_list[0], ndcg_list[0], hr_list[1], ndcg_list[1]))
    print(f'Loss: {avg_loss:.4f}')
    return {'loss': avg_loss, 'HR5': hr_list[0], 'NDCG5': ndcg_list[0],
            'HR10': hr_list[1], 'NDCG10': ndcg_list[1]}

############################################
# Training Function for One Fold (Original Genre Model)
############################################
def train_fold(fold):
    print(f"\n========== Fold {fold} ==========")
    fold_metrics = FoldMetrics(fold)
    train_csv = f"train_fold{fold}.df"
    test_csv = f"test_fold{fold}.df"
    train_df = pd.read_csv(os.path.join(MERGED_DATA_DIR, train_csv))
    test_df = pd.read_csv(os.path.join(MERGED_DATA_DIR, test_csv))
    train_dataset = GenreDataset(train_df)
    test_dataset = GenreDataset(test_df)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
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
    if args.optimizer == 'nadam':
        optimizer = optim.NAdam(model.parameters(), lr=args.lr, eps=args.eps ,weight_decay=args.l2_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, eps=args.eps, weight_decay=args.l2_decay)
    elif args.optimizer == 'adabelief':
        optimizer = AdaBelief(model.parameters(), lr=args.lr, eps=args.eps, weight_decay=args.l2_decay)
    elif args.optimizer == 'lamb':
        optimizer = Lamb(model.parameters(), lr=args.lr, eps=args.eps, weight_decay=args.l2_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, eps=args.eps, weight_decay=args.l2_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=args.factor)
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
                test_met = evaluate(model, diff, f"test_fold{fold}.df", device)
            fold_metrics.add_test_metrics(epoch + 1, test_met)
            scheduler.step()
    os.makedirs("./models", exist_ok=True)
    torch.save(model.state_dict(), f"./models/genre_tenc_fold{fold}.pth")
    torch.save(diff, f"./models/genre_diff_fold{fold}.pth")
    return fold_metrics

############################################
# New Training Function for Experimental Folds (p1 and p2)
############################################
def train_experiment_genre_with_length(exp, seq_length):
    """
    Trains the genre model using experimental folds.
    'exp' should be either "p1" or "p2". This function loads files
    "train_fold{exp}.df" and "test_fold{exp}.df" and builds the model using the provided
    sequence length (i.e. training sequence length, without target).
    """
    print(f"\n========== Experiment {exp} ==========")
    fold_metrics = FoldMetrics(exp)
    train_csv = f"train_fold{exp}.df"
    test_csv = f"test_fold{exp}.df"
    train_df = pd.read_csv(os.path.join(MERGED_DATA_DIR, train_csv))
    test_df = pd.read_csv(os.path.join(MERGED_DATA_DIR, test_csv))
    train_dataset = GenreDataset(train_df)
    test_dataset = GenreDataset(test_df)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # Determine genre vocabulary size from statics if available.
    statics_path = os.path.join(MERGED_DATA_DIR, "statics.csv")
    if os.path.exists(statics_path):
        statics_df = pd.read_csv(statics_path)
        statics = dict(zip(statics_df['statistic'], statics_df['value']))
        genre_vocab_size = int(statics.get("num_genres", 18))
    else:
        genre_vocab_size = 18
    model = Tenc(args.hidden_factor, genre_vocab_size, seq_length, args.dropout_rate, args.diffuser_type, device)
    diff = diffusion(args.timesteps, args.beta_start, args.beta_end, args.w)
    model.to(device)
    if args.optimizer == 'nadam':
        optimizer = optim.NAdam(model.parameters(), lr=args.lr, eps=2e-5, weight_decay=args.l2_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, eps=2e-5, weight_decay=args.l2_decay)
    elif args.optimizer == 'adabelief':
        optimizer = AdaBelief(model.parameters(), lr=args.lr, eps=2e-5, weight_decay=args.l2_decay)
    elif args.optimizer == 'lamb':
        optimizer = Lamb(model.parameters(), lr=args.lr, eps=2e-5, weight_decay=args.l2_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, eps=2e-5, weight_decay=args.l2_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_losses.append(loss.item())
        avg_epoch_loss = np.mean(epoch_losses)
        fold_metrics.add_train_loss(epoch + 1, avg_epoch_loss)
        if args.report_epoch:
            print(f"Experiment {exp} Epoch {epoch + 1:03d}; Train loss: {avg_epoch_loss:.4f}; Time: {Time.strftime('%H:%M:%S', Time.gmtime(Time.time() - start_time))}")
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                print(f"Experiment {exp}: Evaluation at Epoch {epoch + 1}")
                test_metrics = evaluate(model, diff, test_csv, device)
            fold_metrics.add_test_metrics(epoch + 1, test_metrics)
            scheduler.step()
    os.makedirs("./models", exist_ok=True)
    torch.save(model.state_dict(), f"./models/genre_tenc_{exp}.pth")
    torch.save(diff, f"./models/genre_diff_{exp}.pth")
    return fold_metrics

############################################
# New Training Function for Experimental Fold p3
############################################
def train_experiment_genre_p3(seq_length):
    
    print("\n========== Experiment p3 ==========")
    fold_metrics = FoldMetrics("p3")
    train_csv = "train_foldp3.df"
    valid_csv = "valid_foldp3.df"
    test_csv = "test_foldp3.df"
    train_df = pd.read_csv(os.path.join(MERGED_DATA_DIR, train_csv))
    valid_df = pd.read_csv(os.path.join(MERGED_DATA_DIR, valid_csv))
    test_df = pd.read_csv(os.path.join(MERGED_DATA_DIR, test_csv))
    train_dataset = GenreDataset(train_df)
    valid_dataset = GenreDataset(valid_df)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # Determine genre vocabulary size from statics if available.
    statics_path = os.path.join(MERGED_DATA_DIR, "statics.csv")
    if os.path.exists(statics_path):
        statics_df = pd.read_csv(statics_path)
        statics = dict(zip(statics_df['statistic'], statics_df['value']))
        genre_vocab_size = int(statics.get("num_genres", 18))
    else:
        genre_vocab_size = 18
    model = Tenc(args.hidden_factor, genre_vocab_size, seq_length, args.dropout_rate, args.diffuser_type, device)
    diff = diffusion(args.timesteps, args.beta_start, args.beta_end, args.w)
    model.to(device)
    if args.optimizer == 'nadam':
        optimizer = optim.NAdam(model.parameters(), lr=args.lr, eps=2e-5, weight_decay=args.l2_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, eps=2e-5, weight_decay=args.l2_decay)
    elif args.optimizer == 'adabelief':
        optimizer = AdaBelief(model.parameters(), lr=args.lr, eps=2e-5, weight_decay=args.l2_decay)
    elif args.optimizer == 'lamb':
        optimizer = Lamb(model.parameters(), lr=args.lr, eps=2e-5, weight_decay=args.l2_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, eps=2e-5, weight_decay=args.l2_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Training loop using training data; evaluation on validation set during training.
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
            print(f"Experiment p3 Epoch {epoch + 1:03d}; Train loss: {avg_epoch_loss:.4f}; Time: {Time.strftime('%H:%M:%S', Time.gmtime(Time.time() - start_time))}")
        # Evaluate on validation split every 10 epochs.
        if (epoch + 1) % 10 == 0:
            print(f"Experiment p3: Validation Evaluation at Epoch {epoch + 1}")
            _ = evaluate(model, diff, valid_csv, device)
            scheduler.step()
    # Save the trained model
    os.makedirs("./models", exist_ok=True)
    torch.save(model.state_dict(), "./models/genre_tenc_p3.pth")
    torch.save(diff, "./models/genre_diff_p3.pth")
    # Final evaluation on test split after training.
    print("Final Evaluation on p3 test set:")
    test_metrics = evaluate(model, diff, test_csv, device)
    fold_metrics.add_test_metrics(args.epoch, test_metrics)
    return fold_metrics

############################################
# Main Function
############################################
def main():
    NUM_FOLDS = 10
    if args.tune:
        tuning_fold = 1

        lr_candidates = [0.1, 0.05, 0.01, 0.005, 0.001]
        optimizer_candidates = ['nadam', 'lamb', 'adamw', 'adabelief']
        timesteps_candidates =  [50, 100, 150, 200, 250]
        dropout_candidates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 5e-7, 1e-8, 1e-16]
        l2_candidates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 5e-7, 1e-8, 1e-16]
        eps_candidates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 5e-7, 1e-8, 1e-16]
        scheduler_candidates = ['reduce_on_plateau', 'cosine', 'step']
        scheduler_factor_candidates = [0.1, 0.3, 0.5, 0.7, 0.9]
        scheduler_eps_candidates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 5e-7, 1e-8, 1e-16]

        # Create tuning recorders
        tuning_lr_recorder = TuningRecorder("lr", lr_candidates, save_dir="category")
        tuning_optimizer_recorder = TuningRecorder("optimizer", optimizer_candidates, save_dir="category")
        tuning_timesteps_recorder = TuningRecorder("timesteps", timesteps_candidates, save_dir="category")
        tuning_dropout_recorder = TuningRecorder("dropout_rate", dropout_candidates, save_dir="category")
        tuning_l2_recorder = TuningRecorder("l2_decay", l2_candidates, save_dir="category")
        tuning_eps_recorder = TuningRecorder("eps", eps_candidates, save_dir="category")
        tuning_scheduler_recorder = TuningRecorder("scheduler", scheduler_candidates, save_dir="category")
        tuning_scheduler_eps_recorder = TuningRecorder("scheduler_eps", scheduler_eps_candidates, save_dir="category")
        tuning_scheduler_factor = TuningRecorder("scheduler_factor", scheduler_factor_candidates, save_dir="category")
        # Tuning learning rate
        for candidate in tqdm(lr_candidates, desc="Tuning lr"):
            args.lr = candidate
            fm = train_fold(tuning_fold)
            fold_hr10_list = [fm.test_metrics[epoch]['HR10'] for epoch in sorted(fm.test_metrics.keys())]
            for i, hr in enumerate(fold_hr10_list):
                tuning_lr_recorder.record(candidate, (i + 1) * 10, hr)
            print(f"[lr candidate {candidate}] Fold {tuning_fold}: HR@10 = {fold_hr10_list}")
        best_lr = tuning_lr_recorder.find_best()
        print("\nBest learning rate found:", best_lr)
        args.lr = best_lr

        # Tuning optimizer
        for candidate in tqdm(optimizer_candidates, desc="Tuning optimizer"):
            args.optimizer = candidate
            fm = train_fold(tuning_fold)
            fold_hr10_list = [fm.test_metrics[epoch]['HR10'] for epoch in sorted(fm.test_metrics.keys())]
            for i, hr in enumerate(fold_hr10_list):
                tuning_optimizer_recorder.record(candidate, (i + 1) * 10, hr)
            print(f"[optimizer candidate {candidate}] Fold {tuning_fold}: HR@10 = {fold_hr10_list}")
        best_optimizer = tuning_optimizer_recorder.find_best()
        print("\nBest optimizer found:", best_optimizer)
        args.optimizer = best_optimizer

        # Tuning timesteps
        for candidate in tqdm(timesteps_candidates, desc="Tuning timesteps"):
            args.timesteps = candidate
            fm = train_fold(tuning_fold)
            fold_hr10_list = [fm.test_metrics[epoch]['HR10'] for epoch in sorted(fm.test_metrics.keys())]
            for i, hr in enumerate(fold_hr10_list):
                tuning_timesteps_recorder.record(candidate, (i + 1) * 10, hr)
            print(f"[timesteps candidate {candidate}] Fold {tuning_fold}: HR@10 = {fold_hr10_list}")
        best_timesteps = tuning_timesteps_recorder.find_best()
        print("\nBest timesteps found:", best_timesteps)
        args.timesteps = best_timesteps

        # # Tuning dropout_rate
        # for candidate in tqdm(dropout_candidates, desc="Tuning dropout_rate"):
        #     args.dropout_rate = candidate
        #     fm = train_fold(tuning_fold)
        #     fold_hr10_list = [fm.test_metrics[epoch]['HR10'] for epoch in sorted(fm.test_metrics.keys())]
        #     for i, hr in enumerate(fold_hr10_list):
        #         tuning_dropout_recorder.record(candidate, (i + 1) * 10, hr)
        #     print(f"[dropout_rate candidate {candidate}] Fold {tuning_fold}: HR@10 = {fold_hr10_list}")
        # best_dropout = tuning_dropout_recorder.find_best()
        # print("\nBest dropout_rate found:", best_dropout)
        # args.dropout_rate = best_dropout

        # # Tuning l2_decay
        # for candidate in tqdm(l2_candidates, desc="Tuning l2_decay"):
        #     args.l2_decay = candidate
        #     fm = train_fold(tuning_fold)
        #     fold_hr10_list = [fm.test_metrics[epoch]['HR10'] for epoch in sorted(fm.test_metrics.keys())]
        #     for i, hr in enumerate(fold_hr10_list):
        #         tuning_l2_recorder.record(candidate, (i + 1) * 10, hr)
        #     print(f"[l2_decay candidate {candidate}] Fold {tuning_fold}: HR@10 = {fold_hr10_list}")
        # best_l2 = tuning_l2_recorder.find_best()
        # print("\nBest l2_decay found:", best_l2)
        # args.l2_decay = best_l2

        # # Tuning eps
        # for candidate in tqdm(eps_candidates, desc="Tuning eps"):
        #     args.eps = candidate
        #     fm = train_fold(tuning_fold)
        #     fold_hr10_list = [fm.test_metrics[epoch]['HR10'] for epoch in sorted(fm.test_metrics.keys())]
        #     for i, hr in enumerate(fold_hr10_list):
        #         tuning_eps_recorder.record(candidate, (i + 1) * 10, hr)
        #     print(f"[eps candidate {candidate}] Fold {tuning_fold}: HR@10 = {fold_hr10_list}")
        # best_eps = tuning_eps_recorder.find_best()
        # print("\nBest eps found:", best_eps)
        # args.eps = best_eps

        # # Tuning scheduler
        # for candidate in tqdm(scheduler_candidates, desc="Tuning scheduler"):
        #     args.scheduler = candidate
        #     fm = train_fold(tuning_fold)
        #     fold_hr10_list = [fm.test_metrics[epoch]['HR10'] for epoch in sorted(fm.test_metrics.keys())]
        #     for i, hr in enumerate(fold_hr10_list):
        #         tuning_scheduler_recorder.record(candidate, (i + 1) * 10, hr)
        #     print(f"[scheduler candidate {candidate}] Fold {tuning_fold}: HR@10 = {fold_hr10_list}")
        # best_scheduler = tuning_scheduler_recorder.find_best()
        # print("\nBest scheduler found:", best_scheduler)
        # args.scheduler = best_scheduler

        # # If scheduler is reduce_on_plateau or step, tune scheduler_factor and scheduler_eps
        # if args.scheduler in ['reduce_on_plateau', 'step']:
        #     tuning_scheduler_factor_recorder = TuningRecorder("scheduler_factor", scheduler_factor_candidates, save_dir="category")
        #     for candidate in tqdm(scheduler_factor_candidates, desc="Tuning scheduler_factor"):
        #         args.scheduler_factor = candidate
        #         fm = train_fold(tuning_fold)
        #         fold_hr10_list = [fm.test_metrics[epoch]['HR10'] for epoch in sorted(fm.test_metrics.keys())]
        #         for i, hr in enumerate(fold_hr10_list):
        #             tuning_scheduler_factor_recorder.record(candidate, (i + 1) * 10, hr)
        #         print(f"[scheduler_factor candidate {candidate}] Fold {tuning_fold}: HR@10 = {fold_hr10_list}")
        #     best_scheduler_factor = tuning_scheduler_factor_recorder.find_best()
        #     print("\nBest scheduler_factor found:", best_scheduler_factor)
        #     args.scheduler_factor = best_scheduler_factor
        #     tuning_scheduler_factor_recorder.save_to_file("./category/tuning_scheduler_factor.json")
            
            
        # for candidate in tqdm(scheduler_eps_candidates, desc="Tuning scheduler_eps"):
        #     args.scheduler_eps = candidate
        #     fm = train_fold(tuning_fold)
        #     fold_hr10_list = [fm.test_metrics[epoch]['HR10'] for epoch in sorted(fm.test_metrics.keys())]
        #     for i, hr in enumerate(fold_hr10_list):
        #         tuning_scheduler_eps_recorder.record(candidate, (i + 1) * 10, hr)
        #     print(f"[scheduler_eps candidate {candidate}] Fold {tuning_fold}: HR@10 = {fold_hr10_list}")
        # best_scheduler_eps = tuning_scheduler_eps_recorder.find_best()
        # print("\nBest scheduler_eps found:", best_scheduler_eps)
        # args.scheduler_eps = best_scheduler_eps
        # tuning_scheduler_eps_recorder.save_to_file("./category/tuning_scheduler_eps.json")
        
        

        # Save all best candidates
        os.makedirs("./category", exist_ok=True)
        tuning_lr_recorder.save_to_file("./category/tuning_lr.json")
        tuning_optimizer_recorder.save_to_file("./category/tuning_optimizer.json")
        tuning_timesteps_recorder.save_to_file("./category/tuning_timesteps.json")
        # tuning_dropout_recorder.save_to_file("./category/tuning_dropout.json")
        # tuning_l2_recorder.save_to_file("./category/tuning_l2.json")
        # tuning_eps_recorder.save_to_file("./category/tuning_eps.json")
        # tuning_scheduler_recorder.save_to_file("./category/tuning_scheduler.json")

        # Write fold metrics for tuning fold
        with open("./category/fold_metrics_tune.txt", "w") as f:
            f.write("Detailed fold metrics (tuning mode, fold 1):\n")
            f.write(str(train_fold(tuning_fold)))
        print("Tuning fold metrics saved to ./category/fold_metrics_tune.txt")

        best_candidates = {
            "lr": best_lr,
            "optimizer": best_optimizer,
            "timesteps": best_timesteps,
            # "dropout_rate": best_dropout,
            # "l2_decay": best_l2,
            # "eps": best_eps,
            # "scheduler": best_scheduler,
        }
        # if args.scheduler in ['reduce_on_plateau', 'step']:
        #     best_candidates["scheduler_factor"] = args.scheduler_factor
        #     best_candidates["scheduler_eps"] = args.scheduler_eps

        with open("./category/best_candidates.json", "w") as f:
            json.dump(best_candidates, f, indent=2)
        print("Best candidates saved to ./category/best_candidates.json")

    else:
        
        fold_metrics_list = []
        for fold in range(1, NUM_FOLDS + 1):
            fm = train_fold(fold)
            fold_metrics_list.append(fm)
            print("Results for Fold", fold)
            print(fm)
            os.makedirs("./category", exist_ok=True)
        avg_metrics = AverageMetrics()
        for fm in fold_metrics_list:
            avg_metrics.add_fold_metrics(fm)
        avg_metrics.compute_averages()
        print("\n========== Average Metrics Across Folds ==========")
        print(avg_metrics)
        with open("category/average_metrics.txt", "w") as f:
            f.write(str(avg_metrics))
        print("Full average metrics saved to category/average_metrics.txt")
        loss_recorder = LossRecorder(save_dir="category")
        for fm in fold_metrics_list:
            sorted_train = [loss for (ep, loss) in sorted(fm.train_losses, key=lambda x: x[0])]
            sorted_test = [fm.test_metrics[ep]['loss'] for ep in sorted(fm.test_metrics.keys())]
            loss_recorder.add_fold(fm.fold_number, sorted_train, sorted_test)
        loss_recorder.save_to_file("category/loss_data.json")
        avg_train, avg_test = loss_recorder.compute_average_losses()
        epochs_train = sorted(avg_train.keys())
        np.savetxt("category/avg_train_loss.txt", np.array([avg_train[e] for e in epochs_train]))
        print("Average training losses saved to category/avg_train_loss.txt")
        metrics_recorder = MetricsRecorder(save_dir="category")
        for fm in fold_metrics_list:
            metrics_recorder.add_fold(fm)
        metrics_recorder.save_to_file("category/average_test_metrics.txt")
    
    # ----- Run Experimental Folds for Genre Model using extra length argument -----
    # For p1 and p2, default sequence lengths are provided.
    # if args.exp_length is not None:
    #     p1_length = args.exp_length if args.exp_length > 0 else 3
    #     p2_length = args.exp_length if args.exp_length > 0 else 10
    #     p3_length = args.exp_length if args.exp_length > 0 else 20
    # else:
    #     p1_length = 3
    #     p2_length = 10
    #     p3_length = 20

    # print("\n========== Running Experimental Folds for Genre Model ==========")
    # exp_metrics_p1 = train_experiment_genre_with_length("p1", p1_length)
    # exp_metrics_p2 = train_experiment_genre_with_length("p2", p2_length)
    # with open("category/genre_exp_p1_metrics.txt", "w") as f:
    #     f.write(str(exp_metrics_p1))
    # with open("category/genre_exp_p2_metrics.txt", "w") as f:
    #     f.write(str(exp_metrics_p2))
    # print("Experimental fold metrics saved to category/genre_exp_p1_metrics.txt and category/genre_exp_p2_metrics.txt")
    
    # # ----- Run Experimental Fold p3 for Genre Model -----
    # print("\n========== Running Experimental Fold p3 for Genre Model ==========")
    # exp_metrics_p3 = train_experiment_genre_p3(p3_length)
    # with open("category/genre_exp_p3_metrics.txt", "w") as f:
    #     f.write(str(exp_metrics_p3))
    # print("Experimental fold metrics saved to category/genre_exp_p3_metrics.txt")

if __name__ == '__main__':
    main()
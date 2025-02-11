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
from Modules_ori import MovieTenc, MovieDiffusion, Tenc, diffusion, load_genres_predictor
from utility import extract_axis_1, calculate_hit
from recorders import LossRecorder, MetricsRecorder, TuningRecorder, FoldMetrics, AverageMetrics
logging.getLogger().setLevel(logging.INFO)
MERGED_DATA_DIR = "data"

############################################
# Argument Parsing and Setup
############################################

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Item Prediction with Diffusion + Focal Loss using 10-Fold CV on merged data."
    )
    parser.add_argument('--tune', action='store_true', default=False, help='Enable tuning.')
    parser.add_argument('--no-tune', action='store_false', dest='tune', help='Disable tuning.')
    parser.add_argument('--epoch', type=int, default=100, help='Number of epochs per fold.')
    parser.add_argument('--random_seed', type=int, default=100, help='Random seed.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64, help='Embedding/hidden size.')
    parser.add_argument('--timesteps', type=int, default=100, help='Timesteps for diffusion.')
    parser.add_argument('--beta_end', type=float, default=0.02, help='Beta end of diffusion.')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='Beta start of diffusion.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--l2_decay', type=float, default=0.45, help='L2 loss regularization coefficient.')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device id.')
    parser.add_argument('--dropout_rate', type=float, default=0.45, help='Dropout rate.')
    parser.add_argument('--w', type=float, default=2.0, help='Weight used in x_start update inside sampler.')
    parser.add_argument('--p', type=float, default=0.1, help='Probability used in cacu_h for random dropout.')
    parser.add_argument('--report_epoch', type=bool, default=True, help='Whether to report metrics each epoch.')
    parser.add_argument('--diffuser_type', type=str, default='mlp1', help='Type of diffuser network: [mlp1, mlp2].')
    parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer type: [adam, adamw, adagrad, rmsprop].')
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

# =============================================================================
# 5. DATASET DEFINITION
# =============================================================================

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
# 6. LOSS FUNCTION: FocalLoss
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
    movie_seq = eval_data['movie_seq'].apply(lambda x: list(map(int, eval(x)))).tolist()
    movie_len = (eval_data['movie_len'].values if 'movie_len' in eval_data.columns
                 else np.array([len(eval(x)) for x in eval_data['movie_seq'].tolist()]))
    movie_target = eval_data['movie_target'].values
    genre_seq = eval_data['genre_seq'].apply(lambda x: list(map(int, eval(x)))).tolist()
    genre_target = eval_data['genre_target'].values
    for i in range(0, len(movie_seq), batch_size):
        seq_batch = torch.LongTensor(movie_seq[i:i+batch_size]).to(device)
        len_seq_batch = torch.LongTensor(movie_len[i:i+batch_size]).to(device)
        target_batch = torch.LongTensor(movie_target[i:i+batch_size]).to(device)
        genre_seq_batch = torch.LongTensor(genre_seq[i:i+batch_size]).to(device)
        genre_target_batch = torch.LongTensor(genre_target[i:i+batch_size]).to(device)
        x_start = model.cacu_x(target_batch)
        h = model.cacu_h(seq_batch, len_seq_batch, args.p)
        n = torch.randint(0, args.timesteps, (seq_batch.shape[0],), device=device).long()
        n_g = torch.randint(0, genre_diff.timesteps, (seq_batch.shape[0],), device=device).long()
        genre_x_start = genre_model.cacu_x(genre_target_batch)
        genre_h = genre_model.cacu_h(genre_seq_batch, len_seq_batch, args.p)
        _, genre_predicted_x = genre_diff.p_losses(genre_model, genre_x_start, genre_h, n_g, loss_type='l2')
        loss1, predicted_x = diff.p_losses(model, x_start, h, n, genres_embd=genre_predicted_x, loss_type='l2')
        predicted_items = model.decoder(predicted_x)
        focal_loss = FocalLoss(alpha=0.15, gamma=7)
        loss2 = focal_loss(predicted_items, target_batch)
        loss = loss2
        losses.append(loss.item())
        prediction = torch.softmax(predicted_items, dim=-1)
        _, topK = prediction.topk(10, dim=1, largest=True, sorted=True)
        topK = topK.cpu().detach().numpy()
        calculate_hit(target_batch, topK, topk, hit_purchase, ndcg_purchase)
        total_samples += len(target_batch)
    avg_loss = np.mean(losses) if losses else 0.0
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

# =============================================================================
# 8. TRAINING FUNCTION FOR ONE FOLD (with Genre Integration)
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
    from_movie_df = train_df
    train_dataset = MovieDataset(train_df)
    val_dataset = MovieDataset(val_df)
    test_dataset = MovieDataset(test_df)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    statics_file = os.path.join(MERGED_DATA_DIR, "statics.csv")
    if os.path.exists(statics_file):
        statics_df = pd.read_csv(statics_file)
        statics = dict(zip(statics_df["statistic"], statics_df["value"]))
        movie_vocab_size_dynamic = int(statics["num_movies"])
        genre_vocab_size_dynamic = int(statics["num_genres"])
        seq_size = int(statics.get("train_seq_length", 10))
    else:
        movie_vocab_size_dynamic = 4000
        genre_vocab_size_dynamic = 18
        seq_size = 10
    model = MovieTenc(args.hidden_factor, 4000, seq_size, args.dropout_rate, args.diffuser_type, device)
    diff = MovieDiffusion(args.timesteps, args.beta_start, args.beta_end, args.w)
    genre_model = Tenc(args.hidden_factor, genre_vocab_size_dynamic, seq_size, args.dropout_rate,
                        args.diffuser_type, device)
    genre_diff = diffusion(100, args.beta_start, args.beta_end, args.w)
    genre_model, genre_diff = load_genres_predictor(genre_model)
    for param in genre_model.parameters():
        param.requires_grad = False
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
        optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-3, weight_decay=args.l2_decay)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)
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
            focal_loss = FocalLoss(alpha=0.15, gamma=7)
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
                val_dict = evaluate(model, genre_model, genre_diff, val_csv, diff, device)
                test_dict = evaluate(model, genre_model, genre_diff, test_csv, diff, device)
            fold_metrics.add_val_metrics(epoch + 1, val_dict)
            fold_metrics.add_test_metrics(epoch + 1, test_dict)
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

# =============================================================================
# 9. MAIN FUNCTION WITH IF-ELSE (Tuning vs. Full 10-Fold CV)
# =============================================================================
def main():
    NUM_FOLDS = 10
    if args.tune:
        tuning_fold = 1

        # Set initial default values (these will be updated by tuning)
        args.lr = 0.001
        args.optimizer = "adamw"
        args.timesteps = 100

        # Candidate lists for sequential tuning.
        lr_candidates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
        optimizer_candidates = ['adam', 'adamw', 'adagrad', 'rmsprop']
        timesteps_candidates = [i * 100 for i in range(1, 6)]

        # Create TuningRecorder objects with candidate lists and save_dir "item".
        tuning_lr_recorder = TuningRecorder("lr", lr_candidates, save_dir="item")
        tuning_optimizer_recorder = TuningRecorder("optimizer", optimizer_candidates, save_dir="item")
        tuning_timesteps_recorder = TuningRecorder("timesteps", timesteps_candidates, save_dir="item")

        # --- Tune Learning Rate ---
        for candidate in tqdm(lr_candidates, desc="Tuning lr"):
            args.lr = candidate
            fm = train_fold(tuning_fold)
            # Extract HR@10 from test metrics at each evaluation epoch.
            fold_hr10_list = [fm.test_metrics[epoch]['HR10'] for epoch in sorted(fm.test_metrics.keys())]
            for i, hr in enumerate(fold_hr10_list):
                tuning_lr_recorder.record(candidate, (i + 1) * 10, hr)
            print(f"[lr candidate {candidate}] Fold {tuning_fold}: HR@10 = {fold_hr10_list}")
        best_lr = tuning_lr_recorder.find_best()
        print("\nBest learning rate found:", best_lr)
        args.lr = best_lr  # update best learning rate for next stage

        # --- Tune Optimizer (with best learning rate fixed) ---
        for candidate in tqdm(optimizer_candidates, desc="Tuning optimizer"):
            args.optimizer = candidate
            fm = train_fold(tuning_fold)
            fold_hr10_list = [fm.test_metrics[epoch]['HR10'] for epoch in sorted(fm.test_metrics.keys())]
            for i, hr in enumerate(fold_hr10_list):
                tuning_optimizer_recorder.record(candidate, (i + 1) * 10, hr)
            print(f"[optimizer candidate {candidate}] Fold {tuning_fold}: HR@10 = {fold_hr10_list}")
        best_optimizer = tuning_optimizer_recorder.find_best()
        print("\nBest optimizer found:", best_optimizer)
        args.optimizer = best_optimizer  # update best optimizer

        # --- Tune Timesteps (with best lr and optimizer fixed) ---
        for candidate in tqdm(timesteps_candidates, desc="Tuning timesteps"):
            args.timesteps = candidate
            fm = train_fold(tuning_fold)
            fold_hr10_list = [fm.test_metrics[epoch]['HR10'] for epoch in sorted(fm.test_metrics.keys())]
            for i, hr in enumerate(fold_hr10_list):
                tuning_timesteps_recorder.record(candidate, (i + 1) * 10, hr)
            print(f"[timesteps candidate {candidate}] Fold {tuning_fold}: HR@10 = {fold_hr10_list}")
        best_timesteps = tuning_timesteps_recorder.find_best()
        print("\nBest timesteps found:", best_timesteps)
        args.timesteps = best_timesteps  # update best timesteps

        # --- Save and Plot Tuning Results ---
        os.makedirs("./item", exist_ok=True)
        tuning_lr_recorder.save_to_file("./item/tuning_lr.json")
        tuning_optimizer_recorder.save_to_file("./item/tuning_optimizer.json")
        tuning_timesteps_recorder.save_to_file("./item/tuning_timesteps.json")
        tuning_lr_recorder.plot()
        tuning_optimizer_recorder.plot()
        tuning_timesteps_recorder.plot()
        with open("./item/fold_metrics_tune.txt", "w") as f:
            f.write("Detailed fold metrics (tuning mode, fold 1):\n")
            f.write(str(train_fold(tuning_fold)))
        print("Tuning fold metrics saved to ./item/fold_metrics_tune.txt")

        # Save best candidate values to a JSON file.
        best_candidates = {
            "lr": best_lr,
            "optimizer": best_optimizer,
            "timesteps": best_timesteps
        }
        with open("./item/best_candidates.json", "w") as f:
            json.dump(best_candidates, f, indent=2)
        print("Best candidates saved to ./item/best_candidates.json")

    else:
        # --------------------- Full 10-Fold CV Mode ---------------------
        args.lr = 0.001
        args.optimizer = "adamw"
        args.timesteps = 100
        fold_metrics_list = []
        for fold in range(1, NUM_FOLDS + 1):
            fm = train_fold(fold)
            fold_metrics_list.append(fm)
            print("Results for Fold", fold)
            print(fm)
            os.makedirs("./item", exist_ok=True)
            with open(f"./item/fold{fold}_metrics.txt", "w") as f:
                f.write(str(fm))
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
            sorted_val = [fm.val_metrics[ep]['loss'] for ep in sorted(fm.val_metrics.keys())]
            sorted_test = [fm.test_metrics[ep]['loss'] for ep in sorted(fm.test_metrics.keys())]
            loss_recorder.add_fold(fm.fold_number, sorted_train, sorted_val, sorted_test)
        loss_recorder.save_to_file("item/loss_data.json")
        avg_train, avg_val, avg_test = loss_recorder.compute_average_losses()
        epochs_train = sorted(avg_train.keys())
        np.savetxt("item/avg_train_loss.txt", np.array([avg_train[e] for e in epochs_train]))
        print("Average training losses saved to item/avg_train_loss.txt")
        loss_recorder.plot_losses()

        metrics_recorder = MetricsRecorder(save_dir="item")
        for fm in fold_metrics_list:
            metrics_recorder.add_fold(fm)
        metrics_recorder.save_to_file("item/average_test_metrics.txt")
        metrics_recorder.plot_metrics()


if __name__ == '__main__':
    main()

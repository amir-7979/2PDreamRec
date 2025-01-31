import numpy as np
import pandas as pd
import math
import random
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import os
import logging
import time as Time

# Torch Data Utilities
from torch.utils.data import Dataset, DataLoader

# Local utility functions (must be in same directory)
from utility import pad_history, calculate_hit, extract_axis_1
# Local modules (your original multi-head attention, feed-forward, etc.)
from Modules_ori import MultiHeadAttention, PositionwiseFeedForward

logging.getLogger().setLevel(logging.INFO)


##############################################################################
#                          ARGUMENT PARSER & SETUP                           #
##############################################################################
def parse_args():
    parser = argparse.ArgumentParser(description="Run DreamRec with Diffusion + Cross-Entropy.")

    parser.add_argument('--tune', action='store_true', default=False, help='Enable tuning.')
    parser.add_argument('--no-tune', action='store_false', dest='tune', help='Disable tuning.')

    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='yc',
                        help='Dataset name (Arman, yc, ks, zhihu, etc.)')
    parser.add_argument('--random_seed', type=int, default=100,
                        help='Random seed.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--layers', type=int, default=1,
                        help='GRU layers (not used in all code, but present).')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors, i.e. embedding size.')
    parser.add_argument('--timesteps', type=int, default=200,
                        help='Timesteps for diffusion.')
    parser.add_argument('--beta_end', type=float, default=0.02,
                        help='Beta end of diffusion.')
    parser.add_argument('--beta_start', type=float, default=0.0001,
                        help='Beta start of diffusion.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate.')
    parser.add_argument('--l2_decay', type=float, default=1e-4,
                        help='L2 loss reg coef.')
    parser.add_argument('--cuda', type=int, default=0,
                        help='CUDA device.')
    parser.add_argument('--dropout_rate', type=float, default=0.001,
                        help='Dropout rate.')
    parser.add_argument('--w', type=float, default=2.0,
                        help='Weight used in x_start update inside sampler.')
    parser.add_argument('--p', type=float, default=0.1,
                        help='Probability used in cacu_h for random dropout.')
    parser.add_argument('--report_epoch', type=bool, default=True,
                        help='Whether to report metrics each epoch.')
    parser.add_argument('--diffuser_type', type=str, default='mlp1',
                        help='Type of diffuser network: [mlp1, mlp2].')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer type: [adam, adamw, adagrad, rmsprop].')
    parser.add_argument('--beta_sche', nargs='?', default='exp',
                        help='Beta schedule: [linear, exp, cosine, sqrt].')
    parser.add_argument('--descri', type=str, default='',
                        help='Description of the run.')
    return parser.parse_args()


args = parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(args.random_seed)


##############################################################################
#                          BETA SCHEDULE & UTILITIES                         #
##############################################################################
def extract(a, t, x_shape):
    """Helper function to gather the correct index t from a
    schedule (shape [timesteps]) for each item in the batch."""
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def linear_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def exp_beta_schedule(timesteps, beta_min=0.1, beta_max=10):
    x = torch.linspace(1, 2 * timesteps + 1, timesteps)
    betas = 1 - torch.exp(
        - beta_min / timesteps - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps)
    )
    return betas


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """Create a beta schedule that discretizes the given alpha_t_bar function."""
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


##############################################################################
#                 DIFFUSION CLASSES (MovieDiffusion & Base)                  #
##############################################################################
class diffusion():
    """Base diffusion class for 2D or 1D latents with a standard MSE objective."""

    def __init__(self, timesteps, beta_start, beta_end, w):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.w = w

        # Choose schedule
        if args.beta_sche == 'linear':
            self.betas = linear_beta_schedule(timesteps=self.timesteps,
                                              beta_start=self.beta_start,
                                              beta_end=self.beta_end)
        elif args.beta_sche == 'exp':
            self.betas = exp_beta_schedule(timesteps=self.timesteps)
        elif args.beta_sche == 'cosine':
            self.betas = cosine_beta_schedule(timesteps=self.timesteps)
        elif args.beta_sche == 'sqrt':
            self.betas = torch.tensor(betas_for_alpha_bar(
                self.timesteps,
                lambda t: 1 - np.sqrt(t + 0.0001)
            )).float()

        # define alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (
                    1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, denoise_model, x_start, h, t, noise=None, loss_type="l2"):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_x = denoise_model(x_noisy, h, t)

        if loss_type == 'l1':
            loss = F.l1_loss(x_start, predicted_x)
        elif loss_type == 'l2':
            loss = F.mse_loss(x_start, predicted_x)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(x_start, predicted_x)
        else:
            raise NotImplementedError("Unknown loss_type")
        return loss, predicted_x

    @torch.no_grad()
    def p_sample(self, model_forward, model_forward_uncon, x, h, t, t_index):
        x_start = (1 + self.w) * model_forward(x, h, t) - self.w * model_forward_uncon(x, t)
        x_t = x
        model_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, model_forward, model_forward_uncon, h):
        x = torch.randn_like(h)
        for n in reversed(range(0, self.timesteps)):
            x = self.p_sample(
                model_forward,
                model_forward_uncon,
                x,
                h,
                torch.full((h.shape[0],), n, device=h.device, dtype=torch.long),
                n
            )
        return x


class MovieDiffusion(diffusion):
    """Subclass to incorporate an extra 'genres_embd' in the forward pass."""

    def __init__(self, timesteps, beta_start, beta_end, w):
        super().__init__(timesteps, beta_start, beta_end, w)

    def p_losses(self, denoise_model, x_start, h, t, genres_embd, noise=None, loss_type="l2"):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_x = denoise_model(x_noisy, h, t, genres_embd)

        if loss_type == 'l1':
            loss = F.l1_loss(x_start, predicted_x)
        elif loss_type == 'l2':
            loss = F.mse_loss(x_start, predicted_x)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(x_start, predicted_x)
        else:
            raise NotImplementedError("Unknown loss_type")
        return loss, predicted_x

    @torch.no_grad()
    def p_sample(self, model_forward, model_forward_uncon, x, h, t, t_index, genres_embd):
        x_start = (1 + self.w) * model_forward(x, h, t, genres_embd) - self.w * model_forward_uncon(x, t, genres_embd)
        x_t = x
        model_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, model_forward, model_forward_uncon, h, genres_embd):
        x = torch.randn_like(h)
        for n in reversed(range(0, self.timesteps)):
            x = self.p_sample(
                model_forward,
                model_forward_uncon,
                x,
                h,
                torch.full((h.shape[0],), n, device=h.device, dtype=torch.long),
                n,
                genres_embd
            )
        return x


##############################################################################
#                  MODEL CLASSES (Tenc, MovieTenc, etc.)                    #
##############################################################################
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class Tenc(nn.Module):
    """Transformer-based encoder with a simpler diffusion forward pass."""

    def __init__(self, hidden_size, item_num, state_size, dropout, diffuser_type, device, num_heads=1):
        super(Tenc, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.dropout = nn.Dropout(dropout)
        self.diffuser_type = diffuser_type
        self.device = device

        self.item_embeddings = nn.Embedding(num_embeddings=item_num + 1, embedding_dim=hidden_size)
        nn.init.normal_(self.item_embeddings.weight, 0, 1)

        self.none_embedding = nn.Embedding(num_embeddings=1, embedding_dim=self.hidden_size)
        nn.init.normal_(self.none_embedding.weight, 0, 1)

        self.positional_embeddings = nn.Embedding(num_embeddings=state_size, embedding_dim=hidden_size)
        self.emb_dropout = nn.Dropout(dropout)

        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        self.s_fc = nn.Linear(hidden_size, item_num)

        self.step_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
        )
        self.emb_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size * 2)
        )
        self.diff_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
        )

        if self.diffuser_type == 'mlp1':
            self.diffuser = nn.Sequential(
                nn.Linear(self.hidden_size * 3, self.hidden_size)
            )
        elif self.diffuser_type == 'mlp2':
            self.diffuser = nn.Sequential(
                nn.Linear(self.hidden_size * 3, self.hidden_size * 2),
                nn.GELU(),
                nn.Linear(self.hidden_size * 2, self.hidden_size)
            )

    def forward(self, x, h, step):
        """Diffusion forward pass: x_noisy + representation h + time embedding => predicted x."""
        t = self.step_mlp(step)
        cat_in = torch.cat((x, h, t), dim=1)
        return self.diffuser(cat_in)

    def forward_uncon(self, x, step):
        """Unconditional forward pass: same but with a placeholder embedding for h."""
        h = self.none_embedding(torch.tensor([0]).to(self.device))
        h = torch.cat([h.view(1, self.hidden_size)] * x.shape[0], dim=0)
        t = self.step_mlp(step)
        cat_in = torch.cat((x, h, t), dim=1)
        return self.diffuser(cat_in)

    def cacu_x(self, x):
        """Convert item IDs to embeddings."""
        return self.item_embeddings(x)

    def cacu_h(self, states, len_states, p):
        """Compute the hidden representation h from the sequence states."""
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)

        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask

        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        h = state_hidden.squeeze()

        # random dropout in cacu_h
        B, D = h.shape[0], h.shape[1]
        mask1d = (torch.sign(torch.rand(B) - p) + 1) / 2
        maske1d = mask1d.view(B, 1)
        mask2d = torch.cat([maske1d] * D, dim=1).to(self.device)
        none_emb = self.none_embedding(torch.tensor([0]).to(self.device))
        h = h * mask2d + none_emb * (1 - mask2d)
        return h

    def predict(self, states, len_states, diff):
        """Used in the older code for direct sampling (no genres)."""
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)

        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask

        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        h = state_hidden.squeeze()

        x = diff.sample(self.forward, self.forward_uncon, h)
        test_item_emb = self.item_embeddings.weight
        scores = torch.matmul(x, test_item_emb.transpose(0, 1))
        return scores


class MovieTenc(Tenc):
    """MovieTenc uses additional 'genres_embd' in the forward pass."""

    def __init__(self, hidden_size, item_num, state_size, dropout, diffuser_type, device, num_heads=1):
        super(Tenc, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.dropout = nn.Dropout(0.3)
        self.diffuser_type = diffuser_type
        self.device = device
        self.item_embeddings = nn.Embedding(num_embeddings=item_num + 1, embedding_dim=hidden_size)
        nn.init.normal_(self.item_embeddings.weight, 0, 1)
        self.none_embedding = nn.Embedding(num_embeddings=1, embedding_dim=self.hidden_size)
        nn.init.normal_(self.none_embedding.weight, 0, 1)
        self.positional_embeddings = nn.Embedding(num_embeddings=state_size, embedding_dim=hidden_size)
        self.emb_dropout = nn.Dropout(0.3)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        self.s_fc = nn.Linear(hidden_size, item_num)
        self.embedding_dropout = nn.Dropout(0.3)  # Dropout for embeddings

        self.step_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
        )
        self.emb_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size * 2)
        )
        self.diff_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 4, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
        )

        # Diffuser
        if self.diffuser_type == 'mlp1':
            self.diffuser = nn.Sequential(
                nn.Linear(self.hidden_size * 4, self.hidden_size)
            )
            self.decoder = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size * 4),
                nn.ReLU(),
                nn.Linear(self.hidden_size * 4, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.item_num),
            )
        elif self.diffuser_type == 'mlp2':
            self.diffuser = nn.Sequential(
                nn.Linear(self.hidden_size * 4, self.hidden_size * 2),
                nn.GELU(),
                nn.Linear(self.hidden_size * 2, self.hidden_size)
            )
            self.decoder = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size * 2),
                nn.ReLU(),
                nn.Linear(self.hidden_size * 2, self.hidden_size * 4),
                nn.ReLU(),
                nn.Linear(self.hidden_size * 4, self.hidden_size * 2),
                nn.ReLU(),
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.item_num),
            )

    def cacu_x(self, x):
        return self.item_embeddings(x)

    def forward(self, x, h, step, genres_embd):
        t = self.step_mlp(step)
        cat_in = torch.cat((x, h, t, genres_embd), dim=1)
        return self.diffuser(cat_in)

    def forward_uncon(self, x, step, genres_embd):
        h = self.none_embedding(torch.tensor([0]).to(self.device))
        h = torch.cat([h.view(1, self.hidden_size)] * x.shape[0], dim=0)
        t = self.step_mlp(step)
        cat_in = torch.cat((x, h, t, genres_embd), dim=1)
        return self.diffuser(cat_in)

    def predict(self, states, len_states, diff, genres_embd):
        """Predict with final decode (softmax)."""
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        seq *= mask

        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        h = state_hidden.squeeze()

        x = diff.sample(self.forward, self.forward_uncon, h, genres_embd)
        scores = F.softmax(self.decoder(x), dim=-1)
        return scores


##############################################################################
#                  LOAD GENRES MODEL                   #
##############################################################################
def load_genres_predictor(tenc, tenc_path='models/tencVG49.pth', diff_path='models/diffVG49.pth'):
    tenc.load_state_dict(torch.load(tenc_path))
    diff = torch.load(diff_path)
    return tenc, diff


def one_hot_encoding(target, item_num):
    num = target.size()[0]
    encoded_target = torch.zeros(num, item_num)
    for i, row in enumerate(encoded_target):
        row[target[i]] = 1
    return encoded_target


##############################################################################
#                    SIMPLE EARLY STOPPING IMPLEMENTATION                    #
##############################################################################
class EarlyStopper:
    def __init__(self, patience=5, higher_better=True):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.higher_better = higher_better

    def should_stop(self, metric):
        """Return True if we need to stop based on 'metric'."""
        if self.best_score is None:
            self.best_score = metric
            return False

        # Determine if we improved
        improved = (metric > self.best_score) if self.higher_better else (metric < self.best_score)
        if improved:
            self.best_score = metric
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False


##############################################################################
#                       DATASET & DATALOADER FOR TRAINING                    #
##############################################################################
class RecDataset(Dataset):
    """A simple torch dataset for your train_data.df."""

    def __init__(self, dataframe):
        # The dataframe columns must match your usage
        self.seqs = dataframe['seq'].tolist()
        self.len_seqs = dataframe['len_seq'].tolist()
        self.targets = dataframe['next'].tolist()
        self.genre_seqs = dataframe['seq_genres'].tolist()
        self.genre_targets = dataframe['target_genre'].tolist()

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return (torch.tensor(self.seqs[idx], dtype=torch.long),
                torch.tensor(self.len_seqs[idx], dtype=torch.long),
                torch.tensor(self.targets[idx], dtype=torch.long),
                torch.tensor(self.genre_seqs[idx], dtype=torch.long),
                torch.tensor(self.genre_targets[idx], dtype=torch.long))


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Check inputs and targets
        if inputs.ndim == 2 and targets.ndim == 1:
            ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        else:
            raise ValueError("Inputs should be [batch_size, num_classes], and targets should be [batch_size]")

        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


##############################################################################
#                                Loss                                 #
##############################################################################
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        num_classes = pred.size(1)
        smooth_labels = torch.full_like(pred, self.smoothing / (num_classes - 1))
        smooth_labels.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        log_probs = F.log_softmax(pred, dim=1)
        return -(smooth_labels * log_probs).sum(dim=1).mean()


##############################################################################
#                                Loss                                 #
##############################################################################
class NoiseConditionalScoreLoss(nn.Module):
    def __init__(self):
        super(NoiseConditionalScoreLoss, self).__init__()

    def forward(self, predicted_score, true_score, noise_level):
        # Weight loss by noise level
        loss = noise_level * F.mse_loss(predicted_score, true_score)
        return loss


##############################################################################
#                  EVALUATION LOGIC                      #
##############################################################################

def evaluate(model, genre_model, genre_diff, test_data, diff, device):
    eval_data = pd.read_pickle(os.path.join(data_directory, test_data))
    batch_size = 128
    topk = [5, 10, 20]
    total_samples = 0
    hit_purchase = np.zeros(len(topk))
    ndcg_purchase = np.zeros(len(topk))
    losses = []

    # Extract test data sequences and targets
    seq = eval_data['seq'].apply(lambda x: list(map(int, x))).tolist()  # Convert to list of integers
    len_seq = eval_data['len_seq'].values
    target = eval_data['next'].values

    genre_seq = eval_data['seq_genres'].apply(lambda x: list(map(int, x))).tolist()
    genre_len_seq = eval_data['len_seq'].values
    genre_target = eval_data['target_genre'].values

    for i in range(0, len(seq), batch_size):
        seq_batch = torch.LongTensor(seq[i:i + batch_size]).to(device)
        len_seq_batch = torch.LongTensor(len_seq[i:i + batch_size]).to(device)
        target_batch = torch.LongTensor(target[i:i + batch_size]).to(device)

        genre_seq_batch = torch.LongTensor(genre_seq[i:i + batch_size]).to(device)
        genre_target_batch = torch.LongTensor(genre_target[i:i + batch_size]).to(device)

        x_start = model.cacu_x(target_batch)
        h = model.cacu_h(seq_batch, len_seq_batch, args.p)

        n = torch.randint(0, args.timesteps, (h.size(0),), device=device).long()

        # Compute genre embeddings
        n_g = torch.randint(0, 500, (h.size(0),), device=device).long()
        genre_x_start = genre_model.cacu_x(genre_target_batch)
        genre_h = genre_model.cacu_h(genre_seq_batch, len_seq_batch, args.p)
        _, genre_predicted_x = genre_diff.p_losses(genre_model, genre_x_start, genre_h, n_g, loss_type='l2')

        # Predict items
        noise = torch.randn_like(x_start)

        _, predicted_x = diff.p_losses(model, x_start, h, n, genres_embd=genre_predicted_x, noise=noise, loss_type='l2')
        noise_level = (n.float() + 1) / args.timesteps  # Scale to [0, 1]
        predicted_items = model.decoder(predicted_x)
        noise_loss = noise_conditional_loss(predicted_x, noise, noise_level).mean()
        focal_loss = FocalLoss(alpha=0.5, gamma=2)
        cross_entropy_loss = focal_loss(model.decoder(predicted_x), target_batch)
        loss = (args.alpha * noise_loss + (1 - args.alpha) * cross_entropy_loss) / 2
        losses.append(loss.item())

        # Get top-k predictions
        prediction = F.softmax(predicted_items, dim=-1)
        _, topK = prediction.topk(20, dim=1, largest=True, sorted=True)
        topK = topK.cpu().detach().numpy()

        # Calculate hit and NDCG metrics using the new function
        calculate_hit(target_batch, topK, topk, hit_purchase, ndcg_purchase)

        total_samples += len(target_batch)

    # Calculate averages
    avg_loss = np.mean(losses)
    hr_list = hit_purchase / total_samples
    ndcg_list = ndcg_purchase / total_samples

    # Print results
    print(f"Average Loss: {avg_loss:.4f}")
    for idx, k in enumerate(topk):
        print(f"HR@{k}: {hr_list[idx]:.4f}, NDCG@{k}: {ndcg_list[idx]:.4f}")

    return avg_loss, hr_list.tolist(), ndcg_list.tolist()


##############################################################################
#                                MAIN SCRIPT                                 #
##############################################################################
if __name__ == '__main__':
    # Additional hyperparams not in parse_args:
    args.alpha = 0.8  # weighting factor for diffusion vs. cross-entropy

    # Create a list of possible metrics to tune if needed
    if args.tune:
        from collections import defaultdict


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


        metrics = [
            Metric(name='timesteps', values=[i * 100 for i in range(1, 11)]),
            Metric(name='lr', values=[0.1, 0.01, 0.001, 0.0001, 0.00001]),
            Metric(name='optimizer', values=['adam', 'adamw', 'adagrad', 'rmsprop']),
            Metric(name='alpha', values=[i * 0.05 for i in range(1, 21)])
        ]
        best_metrics = []
    else:
        args.lr = .001
        args.optimizer = 'adam'
        args.alpha = 0.8
        metrics = [
            # We test only a single value for timesteps, for example
            # Adjust as needed
            dict(name='timesteps', values=[500]),
        ]
        # we can store them in a simpler structure
        best_metrics = []

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_directory = './data/' + args.data

    # We'll just do a single run if not tuning
    for metric in metrics:
        # Possibly read bestOne from a previous step. Omitted for brevity.

        for value in metric['values']:
            # set the appropriate param
            if metric['name'] == 'lr':
                args.lr = value
            elif metric['name'] == 'optimizer':
                args.optimizer = value
            elif metric['name'] == 'timesteps':
                args.timesteps = value
            elif metric['name'] == 'alpha':
                args.alpha = value

            # read stats
            data_statis = pd.read_pickle(os.path.join(data_directory, 'data_statis.df'))
            seq_size = data_statis['seq_size'][0]
            item_num = data_statis['item_num'][0]

            # read genre stats
            genres_data_statis = pd.read_pickle(os.path.join(data_directory, 'data_statis_g.df'))
            genres_seq_size = genres_data_statis['seq_size'][0]
            genres_item_num = genres_data_statis['item_num'][0]

            # Construct model & diffusion
            model = MovieTenc(args.hidden_factor, item_num, seq_size, args.dropout_rate,
                              args.diffuser_type, device)
            diff = MovieDiffusion(args.timesteps, args.beta_start, args.beta_end, args.w)

            # Load the genre model
            genre_model = Tenc(args.hidden_factor, genres_item_num, genres_seq_size,
                               args.dropout_rate, args.diffuser_type, device)
            genre_diff = diffusion(500, args.beta_start, args.beta_end, args.w)

            # Load pre-trained
            genre_model, genre_diff = load_genres_predictor(genre_model)
            genre_model.eval()
            # freeze genre model
            for param in genre_model.parameters():
                param.requires_grad = False

            # Choose optimizer
            if args.optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-6, weight_decay=args.l2_decay)
            elif args.optimizer == 'adamw':
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-6, weight_decay=args.l2_decay)
            elif args.optimizer == 'adagrad':
                optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, eps=1e-6, weight_decay=args.l2_decay)
            elif args.optimizer == 'rmsprop':
                optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, eps=1e-6, weight_decay=args.l2_decay)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-6, weight_decay=args.l2_decay)

            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

            model.to(device)
            genre_model.to(device)
            train_data = pd.read_pickle(os.path.join(data_directory, 'train_data.df'))
            train_dataset = RecDataset(train_data)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            hr_max = 0
            best_epoch = 0
            early_stopper = EarlyStopper(patience=5, higher_better=True)
            for epoch in range(args.epoch):
                if epoch < 20:
                    alpha = 0.8
                else:
                    alpha = 0.95
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
                    n = torch.randint(0, args.timesteps, (seq_batch.shape[0],), device=device).long()
                    h = model.cacu_h(seq_batch, len_seq_batch, args.p)
                    n_g = torch.randint(0, 500, (seq_batch.shape[0],), device=device).long()
                    genre_x_start = genre_model.cacu_x(genre_target_batch)
                    genre_h = genre_model.cacu_h(genre_seq_batch, len_seq_batch, args.p)
                    _, genre_predicted_x = genre_diff.p_losses(
                        genre_model, genre_x_start, genre_h, n_g, loss_type='l2'
                    )

                    # diffusion + cross-entropy
                    noise = torch.randn_like(x_start)

                    _, predicted_x = diff.p_losses(model, x_start, h, n, genres_embd=genre_predicted_x, noise=noise,
                                                   loss_type='l2')
                    noise_level = (n.float() + 1) / args.timesteps
                    predicted_items = model.decoder(predicted_x)
                    noise_conditional_loss = NoiseConditionalScoreLoss()

                    noise_loss = noise_conditional_loss(predicted_x, noise, noise_level).mean()
                    focal_loss = FocalLoss(alpha=0.5, gamma=2)
                    cross_entropy_loss = focal_loss(model.decoder(predicted_x), target_batch)
                    loss = (args.alpha * noise_loss + (1 - args.alpha) * cross_entropy_loss) / 2
                    loss.backward()
                    optimizer.step()

                # optional: scheduler.step() at each epoch
                # scheduler.step()

                if args.report_epoch:
                    if epoch % 1 == 0:
                        print("Epoch {:03d}; Train loss: {:.4f}; Time: {}".format(
                            epoch, loss.item(),
                            Time.strftime("%H: %M: %S", Time.gmtime(Time.time() - start_time))
                        ))

                if (epoch + 1) % 10 == 0:
                    eval_start = Time.time()
                    model.eval()
                    with torch.no_grad():
                        print('-------------------------- VAL PHASE --------------------------')
                        avg_loss, hr_val, ndcg_val = evaluate(model, genre_model, genre_diff, 'val_data.df', diff,
                                                              device)
                        print('-------------------------- TEST PHASE -------------------------')
                        _, hr_test, _ = evaluate(model, genre_model, genre_diff, 'val_data.df', diff, device)

                        print("Evaluation cost: " + Time.strftime(
                            "%H: %M: %S", Time.gmtime(Time.time() - eval_start))
                              )
                        print('----------------------------------------------------------------')
                        scheduler.step(avg_loss)

                        if early_stopper.should_stop(hr_val):
                            print(f"Early stopping triggered at epoch {epoch}")
                            break

                        if not args.tune:
                            torch.save(model.state_dict(), f"./models/tencV{epoch}.pth")
                            torch.save(diff, f"./models/diffV{epoch}.pth")
    print("All done!")
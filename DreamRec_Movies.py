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
from utility import pad_history, calculate_hit, extract_axis_1
from collections import Counter
from Modules_ori import *

logging.getLogger().setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument('--tune', action='store_true', default=False, help='Enable tuning.')
    parser.add_argument('--no-tune', action='store_false', dest='tune', help='Disable tuning.')

    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='yc',
                        help='Amir, yc, ks, zhihu')
    parser.add_argument('--random_seed', type=int, default=100,
                        help='random seed')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', type=int, default=1,
                        help='gru_layers')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--timesteps', type=int, default=200,
                        help='timesteps for diffusion')
    parser.add_argument('--beta_end', type=float, default=0.02,
                        help='beta end of diffusion')
    parser.add_argument('--beta_start', type=float, default=0.0001,
                        help='beta start of diffusion')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate.')
    parser.add_argument('--l2_decay', type=float, default=0,
                        help='l2 loss reg coef.')
    parser.add_argument('--cuda', type=int, default=0,
                        help='cuda device.')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='dropout ')
    parser.add_argument('--w', type=float, default=2.0,
                        help='dropout ')
    parser.add_argument('--p', type=float, default=0.1,
                        help='dropout ')
    parser.add_argument('--report_epoch', type=bool, default=True,
                        help='report frequency')
    parser.add_argument('--diffuser_type', type=str, default='mlp1',
                        help='type of diffuser.')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='type of optimizer.')
    parser.add_argument('--beta_sche', nargs='?', default='exp',
                        help='')
    parser.add_argument('--descri', type=str, default='',
                        help='description of the work.')
    return parser.parse_args()


args = parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(args.random_seed)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def linear_beta_schedule(timesteps, beta_start, beta_end):
    beta_start = beta_start
    beta_end = beta_end
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
    betas = 1 - torch.exp(- beta_min / timesteps - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps))
    return betas


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class diffusion():
    def __init__(self, timesteps, beta_start, beta_end, w):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.w = w

        if args.beta_sche == 'linear':
            self.betas = linear_beta_schedule(timesteps=self.timesteps, beta_start=self.beta_start,
                                              beta_end=self.beta_end)
        elif args.beta_sche == 'exp':
            self.betas = exp_beta_schedule(timesteps=self.timesteps)
        elif args.beta_sche == 'cosine':
            self.betas = cosine_beta_schedule(timesteps=self.timesteps)
        elif args.beta_sche == 'sqrt':
            self.betas = torch.tensor(betas_for_alpha_bar(self.timesteps, lambda t: 1 - np.sqrt(t + 0.0001), )).float()

        # define alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (
                    1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        # print(self.betas)
        if noise is None:
            noise = torch.randn_like(x_start)
            # noise = torch.randn_like(x_start) / 100
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, denoise_model, x_start, h, t, noise=None, loss_type="l2"):
        #
        if noise is None:
            noise = torch.randn_like(x_start)
            # noise = torch.randn_like(x_start) / 100

        #
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        predicted_x = denoise_model(x_noisy, h, t)

        #
        if loss_type == 'l1':
            loss = F.l1_loss(x_start, predicted_x)
        elif loss_type == 'l2':
            loss = F.mse_loss(x_start, predicted_x)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(x_start, predicted_x)
        else:
            raise NotImplementedError()

        return loss, predicted_x

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    @torch.no_grad()
    def p_sample(self, model_forward, model_forward_uncon, x, h, t, t_index):

        x_start = (1 + self.w) * model_forward(x, h, t) - self.w * model_forward_uncon(x, t)
        x_t = x
        model_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
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
        # x = torch.randn_like(h) / 100

        for n in reversed(range(0, self.timesteps)):
            x = self.p_sample(model_forward, model_forward_uncon, x, h,
                              torch.full((h.shape[0],), n, device=device, dtype=torch.long), n)

        return x


class MovieDiffusion(diffusion):
    def __init__(self, timesteps, beta_start, beta_end, w):
        super().__init__(timesteps, beta_start, beta_end, w)

    def p_losses(self, denoise_model, x_start, h, t, genres_embd, noise=None, loss_type="l2"):
        #
        if noise is None:
            noise = torch.randn_like(x_start)
            # noise = torch.randn_like(x_start) / 100

        #
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        predicted_x = denoise_model(x_noisy, h, t, genres_embd)

        #
        if loss_type == 'l1':
            loss = F.l1_loss(x_start, predicted_x)
        elif loss_type == 'l2':
            loss = F.mse_loss(x_start, predicted_x)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(x_start, predicted_x)
        else:
            raise NotImplementedError()

        return loss, predicted_x

    @torch.no_grad()
    def p_sample(self, model_forward, model_forward_uncon, x, h, t, t_index, genres_embd):

        x_start = (1 + self.w) * model_forward(x, h, t, genres_embd) - self.w * model_forward_uncon(x, t, genres_embd)
        x_t = x
        model_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
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
        # x = torch.randn_like(h) / 100

        for n in reversed(range(0, self.timesteps)):
            x = self.p_sample(model_forward, model_forward_uncon, x, h,
                              torch.full((h.shape[0],), n, device=device, dtype=torch.long), n, genres_embd)

        return x


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
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Tenc(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, dropout, diffuser_type, device, num_heads=1):
        super(Tenc, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.dropout = nn.Dropout(dropout)
        self.diffuser_type = diffuser_type
        self.device = device
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=hidden_size,
        )

        nn.init.normal_(self.item_embeddings.weight, 0, 1)
        self.none_embedding = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.hidden_size,
        )
        nn.init.normal_(self.none_embedding.weight, 0, 1)
        self.positional_embeddings = nn.Embedding(
            num_embeddings=state_size,
            embedding_dim=hidden_size
        )
        # emb_dropout is added
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        self.s_fc = nn.Linear(hidden_size, item_num)
        # self.ac_func = nn.ReLU()

        # self.step_embeddings = nn.Embedding(
        #     num_embeddings=50,
        #     embedding_dim=hidden_size
        # )

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

        t = self.step_mlp(step)

        if self.diffuser_type == 'mlp1':
            res = self.diffuser(torch.cat((x, h, t), dim=1))
        elif self.diffuser_type == 'mlp2':
            res = self.diffuser(torch.cat((x, h, t), dim=1))
        return res

    def forward_uncon(self, x, step):
        h = self.none_embedding(torch.tensor([0]).to(self.device))
        h = torch.cat([h.view(1, args.hidden_factor)] * x.shape[0], dim=0)

        t = self.step_mlp(step)

        if self.diffuser_type == 'mlp1':
            res = self.diffuser(torch.cat((x, h, t), dim=1))
        elif self.diffuser_type == 'mlp2':
            res = self.diffuser(torch.cat((x, h, t), dim=1))

        return res

        # return x

    def cacu_x(self, x):
        x = self.item_embeddings(x)
        return x

    def cacu_h(self, states, len_states, p):
        # hidden
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

        B, D = h.shape[0], h.shape[1]
        mask1d = (torch.sign(torch.rand(B) - p) + 1) / 2
        maske1d = mask1d.view(B, 1)
        mask = torch.cat([maske1d] * D, dim=1)
        mask = mask.to(self.device)

        # print(h.device, self.none_embedding(torch.tensor([0]).to(self.device)).device, mask.device)
        h = h * mask + self.none_embedding(torch.tensor([0]).to(self.device)) * (1 - mask)

        return h

    def predict(self, states, len_states, diff):
        # hidden
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
    def __init__(self, hidden_size, item_num, state_size, dropout, diffuser_type, device, num_heads=1):
        super(Tenc, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.dropout = nn.Dropout(dropout)
        self.diffuser_type = diffuser_type
        self.device = device
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 1)
        self.none_embedding = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.hidden_size,
        )
        nn.init.normal_(self.none_embedding.weight, 0, 1)
        self.positional_embeddings = nn.Embedding(
            num_embeddings=state_size,
            embedding_dim=hidden_size
        )
        # emb_dropout is added
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        self.s_fc = nn.Linear(hidden_size, item_num)
        # self.ac_func = nn.ReLU()

        # self.step_embeddings = nn.Embedding(
        #     num_embeddings=50,
        #     embedding_dim=hidden_size
        # )

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
                # nn.Softmax(dim=-1),
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
                # nn.Softmax(dim=-1),
            )

    def cacu_x(self, x):
        x = self.item_embeddings(x)
        return x

    def forward(self, x, h, step, genres_embd):

        t = self.step_mlp(step)

        if self.diffuser_type == 'mlp1':
            res = self.diffuser(torch.cat((x, h, t, genres_embd), dim=1))
        elif self.diffuser_type == 'mlp2':
            res = self.diffuser(torch.cat((x, h, t, genres_embd), dim=1))
        return res

    def forward_uncon(self, x, step, genres_embd):
        h = self.none_embedding(torch.tensor([0]).to(self.device))
        h = torch.cat([h.view(1, 64)] * x.shape[0], dim=0)

        t = self.step_mlp(step)

        if self.diffuser_type == 'mlp1':
            res = self.diffuser(torch.cat((x, h, t, genres_embd), dim=1))
        elif self.diffuser_type == 'mlp2':
            res = self.diffuser(torch.cat((x, h, t, genres_embd), dim=1))

        return res

    def predict(self, states, len_states, diff, genres_embd):
        # hidden
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

        # test_item_emb = self.item_embeddings.weight
        # # scores = torch.matmul(x, test_item_emb.transpose(0, 1))
        # scores = torch.matmul(x / x.norm(dim=-1, keepdim=True),
        #               (test_item_emb / test_item_emb.norm(dim=-1, keepdim=True)).transpose(0, 1))

        return scores


def load_genres_predictor(tenc, tenc_path='models/tencVG79.pth', diff_path='models/diffVG79.pth'):
    tenc.load_state_dict(torch.load(tenc_path))
    diff = torch.load(diff_path)

    return tenc, diff


def one_hot_encoding(target, item_num):
    num = target.size()[0]
    encoded_target = torch.zeros(num, item_num)
    for i, row in enumerate(encoded_target):
        row[target[i]] = 1
    return encoded_target


def evaluate(model, genre_model, genre_diff, test_data, diff, device):
    eval_data = pd.read_pickle(os.path.join(data_directory, test_data))

    batch_size = 100
    evaluated = 0
    total_clicks = 1.0
    total_purchase = 0.0
    total_reward = [0, 0, 0, 0]
    hit_clicks = [0, 0, 0, 0]
    ndcg_clicks = [0, 0, 0, 0]
    hit_purchase = [0, 0, 0, 0]
    ndcg_purchase = [0, 0, 0, 0]

    seq, len_seq, target = list(eval_data['seq'].values), list(eval_data['len_seq'].values), list(
        eval_data['next'].values)
    genre_seq, genre_len_seq, genre_target = list(eval_data['seq_genres'].values), list(
        eval_data['len_seq'].values), list(eval_data['target_genre'].values)

    num_total = len(seq)
    losses = []
    for i in range(num_total // batch_size):
        seq_b, len_seq_b, target_b = seq[i * batch_size: (i + 1) * batch_size], len_seq[i * batch_size: (
                                                                                                                    i + 1) * batch_size], target[
                                                                                                                                          i * batch_size: (
                                                                                                                                                                      i + 1) * batch_size]
        genre_seq_b, genre_len_seq_b, genre_target_b = genre_seq[i * batch_size: (i + 1) * batch_size], genre_len_seq[
                                                                                                        i * batch_size: (
                                                                                                                                    i + 1) * batch_size], genre_target[
                                                                                                                                                          i * batch_size: (
                                                                                                                                                                                      i + 1) * batch_size]

        states = np.array(seq_b)
        states = torch.LongTensor(states)
        states = states.to(device)

        """"""
        seq_t = torch.LongTensor(seq_b)
        len_seq_t = torch.LongTensor(len_seq_b)
        target_t = torch.LongTensor(target_b)

        seq_t = seq_t.to(device)
        target_t = target_t.to(device)
        len_seq_t = len_seq_t.to(device)

        x_start = model.cacu_x(target_t)

        h = model.cacu_h(seq_t, len_seq_t, args.p)
        """"""

        """Add genres data to specified device"""
        genre_seq_b = torch.LongTensor(genre_seq_b)
        genre_len_seq_b = torch.LongTensor(genre_len_seq_b)
        genre_target_b = torch.LongTensor(genre_target_b)

        genre_seq_b = genre_seq_b.to(device)
        genre_target_b = genre_target_b.to(device)
        genre_len_seq_b = genre_len_seq_b.to(device)

        n = torch.randint(0, args.timesteps, (batch_size,), device=device).long()
        n_g = torch.randint(0, 600, (batch_size,), device=device).long()

        genre_x_start = genre_model.cacu_x(genre_target_b)
        genre_h = genre_model.cacu_h(genre_seq_b, genre_len_seq_b, args.p)
        _, genre_predicted_x = genre_diff.p_losses(genre_model, genre_x_start, genre_h, n_g, loss_type='l2')
        """"""

        loss, predicted_x = diff.p_losses(model, x_start, h, n, genres_embd=genre_predicted_x, loss_type='l2')

        predicted_items = model.decoder(predicted_x)
        # loss = loss_function(predicted_items, target_t)

        losses.append(loss.item())

        # prediction = model.predict(states, np.array(len_seq_b), diff, genre_predicted_x)
        # assert False, (np.shape(prediction,), np.shape(predicted_x))
        try:
            # prediction = model.predict(states, np.array(len_seq_b), diff, genre_predicted_x)
            prediction = F.softmax(predicted_items, dim=-1)
            _, topK = prediction.topk(20, dim=1, largest=True, sorted=True)
            topK = topK.cpu().detach().numpy()
            sorted_list2 = np.flip(topK, axis=1)
            sorted_list2 = sorted_list2
            calculate_hit(sorted_list2, topk, target_b, hit_purchase, ndcg_purchase)
        except:
            pass

        total_purchase += batch_size

    hr_list = []
    ndcg_list = []

    print('{:<10s} {:<10s} {:<10s} {:<10s} {:<10s} {:<10s}'.format('HR@' + str(topk[0]), 'NDCG@' + str(topk[0]),
                                                                   'HR@' + str(topk[1]), 'NDCG@' + str(topk[1]),
                                                                   'HR@' + str(topk[2]), 'NDCG@' + str(topk[2])))
    for i in range(len(topk)):
        hr_purchase = hit_purchase[i] / total_purchase
        ng_purchase = ndcg_purchase[i] / total_purchase
        # assert False, ndcg_purchase
        try:
            hr_list.append(hr_purchase)
            # assert False, ng_purchase
            ndcg_list.append(ng_purchase[0, 0])
        except:
            pass

        if i == 1:
            hr_20 = hr_purchase

    if len(hr_list) == 3 and len(ndcg_list) == 3:
        print(
            '{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[0], (ndcg_list[0]), hr_list[1],
                                                                                 (ndcg_list[1]), hr_list[2],
                                                                                 (ndcg_list[2])))

    else:
        pass
    print(f'loss:{sum(losses) / len(losses)}')

    return sum(losses) / len(losses), hr_list[0]


import numpy as np
from tqdm import tqdm
from collections import defaultdict


class Metric:
    def __init__(self, name, values, ):
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


if __name__ == '__main__':
    args.alpha = 0.45
    if args.tune:
        metrics = [
            Metric(name='optimizer', values=['adagrad','adam', 'adamw', 'rmsprop']),
            Metric(name='lr', values=[0.1, 0.01, 0.001, 0.0001, 0.00001]),
            Metric(name='timesteps', values=[i * 100 for i in range(1, 11)]),
            Metric(name='alpha', values=[i * 0.05 for i in range(1, 21)]),
        ]
        best_metrics = list()
    else:
        metrics = [
            Metric(name='lr', values=[0.01]),
            Metric(name='optimizer', values=['adamw']),
            Metric(name='timesteps', values=[100]),
            Metric(name='alpha', values=[0.45]),
        ]

    for metric in metrics:
        for b_m in metrics:
            if b_m.bestOne is not None:
                if b_m.name == 'timesteps':
                    args.timesteps = b_m.bestOne
                    print(f'Timesteps: {args.timesteps}')
                elif b_m.name == 'lr':
                    args.lr = b_m.bestOne
                    print(f'Learning Rate: {args.lr}')
                elif b_m.name == 'optimizer':
                    args.optimizer = b_m.bestOne
                    print(f'Optimizer: {args.optimizer}')
                elif b_m.name == 'alpha':
                    args.alpha = b_m.bestOne
                    print(f'Alpha: {args.alpha}')

        for value in tqdm(metric.values):
            if metric.name == 'lr':
                args.lr = value
            elif metric.name == 'optimizer':
                args.optimizer = value
            elif metric.name == 'timesteps':
                args.timesteps = value
            elif metric.name == 'alpha':
                args.alpha = value

            # args = parse_args()
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

            data_directory = './data/' + args.data
            data_statis = pd.read_pickle(
                os.path.join(data_directory,
                             'data_statis.df'))  # read data statistics, includeing seq_size and item_num
            seq_size = data_statis['seq_size'][0]  # the length of history to define the seq
            item_num = data_statis['item_num'][0]  # total number of items

            """ Load Genres' Meta data"""
            genres_data_statis = pd.read_pickle(
                os.path.join(data_directory, 'data_statis_g.df'))

            genres_seq_size = genres_data_statis['seq_size'][0]
            genres_item_num = genres_data_statis['item_num'][0]
            """"""
            topk = [5, 10, 20]

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            timesteps = args.timesteps

            # args.hidden_factor = 2048
            model = MovieTenc(args.hidden_factor, item_num, seq_size, args.dropout_rate, args.diffuser_type, device)
            diff = MovieDiffusion(args.timesteps, args.beta_start, args.beta_end, args.w)

            """Load Genres' Models"""
            genre_model = Tenc(args.hidden_factor, genres_item_num, genres_seq_size, args.dropout_rate,
                               args.diffuser_type, device)
            genre_diff = diffusion(600, args.beta_start, args.beta_end, args.w)
            genre_model, genre_diff = load_genres_predictor(genre_model)
            genre_model.eval()

            for parameter in genre_model.parameters():
                parameter.requires_grad = False

            """"""
            model.to(device)

            if args.optimizer == 'adagrad':
                optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
            elif args.optimizer == 'adamw':
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
            elif args.optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
            elif args.optimizer == 'rmsprop':
                optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)

            # scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1, total_iters=20)


            """Load the genres model into the specified device"""
            genre_model.to(device)
            """"""
            # optimizer.to(device)

            train_data = pd.read_pickle(os.path.join(data_directory, 'train_data.df'))

            total_step = 0
            hr_max = 0
            best_epoch = 0

            # Loss function
            loss_function = nn.CrossEntropyLoss()

            num_rows = train_data.shape[0]
            num_batches = int(num_rows / args.batch_size)
            for i in range(args.epoch):
                start_time = Time.time()
                for j in range(num_batches):
                    batch = train_data.sample(n=args.batch_size).to_dict()
                    seq = list(batch['seq'].values())
                    len_seq = list(batch['len_seq'].values())
                    target = list(batch['next'].values())

                    """Get data related to genres from the batch"""
                    genre_seq = list(batch['seq_genres'].values())
                    genre_len_seq = list(batch['len_seq'].values())
                    genre_target = list(batch['target_genre'].values())
                    """"""

                    optimizer.zero_grad()
                    seq = torch.LongTensor(seq)
                    len_seq = torch.LongTensor(len_seq)
                    target = torch.LongTensor(target)

                    seq = seq.to(device)
                    target = target.to(device)
                    len_seq = len_seq.to(device)

                    """Add genres data to specified device"""
                    genre_seq = torch.LongTensor(genre_seq)
                    genre_len_seq = torch.LongTensor(genre_len_seq)
                    genre_target = torch.LongTensor(genre_target)

                    genre_seq = genre_seq.to(device)
                    genre_target = genre_target.to(device)
                    genre_len_seq = genre_len_seq.to(device)
                    """"""

                    x_start = model.cacu_x(target).to(device)

                    n = torch.randint(0, args.timesteps, (args.batch_size,), device=device).long()
                    n_g = torch.randint(0, 600, (args.batch_size,), device=device).long()
                    h = model.cacu_h(seq, len_seq, args.p).to(device)

                    """Calculate x_start for genres"""
                    genre_x_start = genre_model.cacu_x(genre_target).to(device)
                    genre_h = genre_model.cacu_h(genre_seq, genre_len_seq, args.p).to(device)
                    _, genre_predicted_x = genre_diff.p_losses(genre_model, genre_x_start, genre_h, n_g, loss_type='l2')

                    """"""

                    loss1, predicted_x = diff.p_losses(model, x_start, h, n, genres_embd=genre_predicted_x,
                                                       loss_type='l2')
                    # loss.backward()
                    # optimizer.step()

                    # encoded_target = one_hot_encoding(target, item_num).to(device)
                    # assert False, model.predict(seq, len_seq, diff, genre_predicted_x).max(dim=1)
                    predicted_items = model.decoder(predicted_x)
                    loss2 = loss_function(predicted_items, target)

                    loss = (args.alpha * loss1) + ((1 - args.alpha) * loss2)

                    loss.backward()
                    optimizer.step()

                    # _ = evaluate(model, genre_model, genre_diff, 'val_data.df', diff, device)

                # scheduler.step()
                if args.report_epoch:
                    if i % 1 == 0:
                        print("Epoch {:03d}; ".format(i) + 'Train loss: {:.4f}; '.format(
                            loss) + "Time cost: " + Time.strftime(
                            "%H: %M: %S", Time.gmtime(Time.time() - start_time)))

                    if (i + 1) % 10 == 0:

                        eval_start = Time.time()
                        print('-------------------------- VAL PHRASE --------------------------')
                        _, hr_val = evaluate(model, genre_model, genre_diff, 'val_data.df', diff, device)
                        print('-------------------------- TEST PHRASE -------------------------')
                        _, hr_test = evaluate(model, genre_model, genre_diff, 'test_data.df', diff, device)
                        print("Evalution cost: " + Time.strftime("%H: %M: %S", Time.gmtime(Time.time() - eval_start)))
                        print('----------------------------------------------------------------')

                        metric.eval_dict[value].append(hr_val)

                        if not args.tune:
                            torch.save(model.state_dict(), f"./models/tencV{i}.pth")
                            torch.save(diff, f"./models/diffV{i}.pth")

        if args.tune:
            metric.find_max_one()
            best_metrics.append(metric)
            torch.save(best_metrics, './tune/metrics_m.dict')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from utility import extract_axis_1, calculate_hit

def load_genres_predictor(tenc, 
                          tenc_path='models/genre_tenc_fold1.pth', 
                          diff_path='models/genre_diff_fold1.pth'):
    """
    Loads a pretrained checkpoint for the genre predictor model.
    For keys where the shape does not match the current model (e.g., due to using mlp2),
    those parameters are skipped.
    """
    # Load the checkpoint state dict.
    checkpoint = torch.load(tenc_path, map_location='cpu', weights_only=True)
    model_dict = tenc.state_dict()

    # Filter out keys with mismatched shapes.
    filtered_checkpoint = {}
    for key, value in checkpoint.items():
        if key in model_dict:
            if value.size() == model_dict[key].size():
                filtered_checkpoint[key] = value
            else:
                print(f"Skipping parameter '{key}': checkpoint shape {value.size()} vs. model shape {model_dict[key].size()}")
        else:
            print(f"Parameter '{key}' not found in the current model.")
    
    # Load the filtered state dict.
    tenc.load_state_dict(filtered_checkpoint, strict=False)
    
    # Load the diffusion checkpoint normally.
    diff = torch.load(diff_path, map_location='cpu', weights_only=True)
    return tenc, diff

############################################
# Helper functions for beta schedules, etc.
############################################
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def linear_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def exp_beta_schedule(timesteps, beta_min=0.1, beta_max=10):
    x = torch.linspace(1, 2 * timesteps + 1, timesteps)
    betas = 1 - torch.exp(- beta_min / timesteps - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps))
    return betas


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


############################################
# Diffusion and MovieDiffusion Classes
############################################

class diffusion():
    def __init__(self, timesteps, beta_start, beta_end, w):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.w = w

        self.betas = linear_beta_schedule(timesteps=self.timesteps, beta_start=self.beta_start,
                                          beta_end=self.beta_end)

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
                              torch.full((h.shape[0],), n, device=h.device, dtype=torch.long), n)

        return x


class MovieDiffusion(diffusion):
    def __init__(self, timesteps, beta_start, beta_end, w):
        super().__init__(timesteps, beta_start, beta_end, w)

    def p_losses(self, denoise_model, x_start, h, t, genres_embd, noise=None, loss_type="l2"):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
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
        model_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + \
                     extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, model_forward, model_forward_uncon, h, genres_embd):
        x = torch.randn_like(h)
        for n in reversed(range(self.timesteps)):
            x = self.p_sample(model_forward, model_forward_uncon, x, h,
                              torch.full((h.shape[0],), n, device=h.device, dtype=torch.long), n, genres_embd)
        return x


############################################
# Model Classes
############################################
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
        h = torch.cat([h.view(1, 64)] * x.shape[0], dim=0)

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


# ------------------------


def get_top_genres(genre_model, diff, genre_seq, len_seq, device, top_k=12):
    scores = genre_model.predict(genre_seq, len_seq, diff)  # shape: [batch_size, num_genres]
    scores = torch.softmax(scores, dim=-1)
    top_genres = torch.topk(scores, top_k, dim=1).indices  # shape: [batch_size, top_k]
    return top_genres




def filter_movie_scores(movie_scores, top_genres, genre_movie_mapping, target_batch):
    batch_size, num_movies = movie_scores.shape
    result = movie_scores.clone()

    # Convert genre_movie_mapping to a boolean mask
    genre_mask = torch.zeros((len(genre_movie_mapping), num_movies), dtype=torch.bool, device=movie_scores.device)

    for genre, movies in genre_movie_mapping.items():
        if genre.isdigit():
            genre_idx = int(genre)
            if genre_idx >= len(genre_movie_mapping):  # Ensure valid index
                continue
            valid_movies = [m for m in movies if 0 <= m < num_movies]  # Ensure valid movies
            if valid_movies:
                genre_mask[genre_idx, valid_movies] = True

    for i in range(batch_size):
        genre_indices = top_genres[i].long()  # Convert to long for indexing
        valid_movies_mask = torch.any(genre_mask[genre_indices], dim=0)

        # Ensure the target movie is included
        target_movie_idx = target_batch[i].item()
        if 0 <= target_movie_idx < num_movies:  # Ensure valid target movie index
            valid_movies_mask[target_movie_idx] = True

        # Zero out invalid movie scores
        result[i] *= valid_movies_mask

    return result




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

    def predict(self, states, len_states, h, diff, genres_embd):
        # hidden
        inputs_emb = self.item_embeddings(states)
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
        #seq *= mask
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        #ff_out *= mask
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out, len_states - 1)
        h = state_hidden.squeeze()

        x = diff.sample(self.forward, self.forward_uncon, h, genres_embd)
        # scores = F.softmax(self.decoder(x), dim=-1)

        # test_item_emb = self.item_embeddings.weight
        # # scores = torch.matmul(x, test_item_emb.transpose(0, 1))
        # scores = torch.matmul(x / x.norm(dim=-1, keepdim=True),
        #               (test_item_emb / test_item_emb.norm(dim=-1, keepdim=True)).transpose(0, 1))

        combined = h + x
        combined = F.relu(combined)

        # Decode the combined representation to produce final recommendation scores.
        scores = self.decoder(combined)
        return scores


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_units, num_heads, dropout_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0

        self.linear_q = nn.Linear(hidden_size, num_units)
        self.linear_k = nn.Linear(hidden_size, num_units)
        self.linear_v = nn.Linear(hidden_size, num_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries, keys):
        """
        :param queries: A 3d tensor with shape of [N, T_q, C_q]
        :param keys: A 3d tensor with shape of [N, T_k, C_k]

        :return: A 3d tensor with shape of (N, T_q, C)

        """
        Q = self.linear_q(queries)  # (N, T_q, C)
        K = self.linear_k(keys)  # (N, T_k, C)
        V = self.linear_v(keys)  # (N, T_k, C)

        # Split and Concat
        split_size = self.hidden_size // self.num_heads
        Q_ = torch.cat(torch.split(Q, split_size, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.split(K, split_size, dim=2), dim=0)  # (h*N, T_k, C/h)
        V_ = torch.cat(torch.split(V, split_size, dim=2), dim=0)  # (h*N, T_k, C/h)

        # Multiplication
        matmul_output = torch.bmm(Q_, K_.transpose(1, 2)) / self.hidden_size ** 0.5  # (h*N, T_q, T_k)

        # Key Masking
        key_mask = torch.sign(torch.abs(keys.sum(dim=-1))).repeat(self.num_heads, 1)  # (h*N, T_k)
        key_mask_reshaped = key_mask.unsqueeze(1).repeat(1, queries.shape[1], 1)  # (h*N, T_q, T_k)
        key_paddings = torch.ones_like(matmul_output) * (-2 ** 32 + 1)
        matmul_output_m1 = torch.where(torch.eq(key_mask_reshaped, 0), key_paddings, matmul_output)  # (h*N, T_q, T_k)

        # Causality - Future Blinding
        diag_vals = torch.ones_like(matmul_output[0, :, :])  # (T_q, T_k)
        tril = torch.tril(diag_vals)  # (T_q, T_k)
        causality_mask = tril.unsqueeze(0).repeat(matmul_output.shape[0], 1, 1)  # (h*N, T_q, T_k)
        causality_paddings = torch.ones_like(causality_mask) * (-2 ** 32 + 1)
        matmul_output_m2 = torch.where(torch.eq(causality_mask, 0), causality_paddings,
                                       matmul_output_m1)  # (h*N, T_q, T_k)

        # Activation
        matmul_output_sm = self.softmax(matmul_output_m2)  # (h*N, T_q, T_k)

        # Query Masking
        query_mask = torch.sign(torch.abs(queries.sum(dim=-1))).repeat(self.num_heads, 1)  # (h*N, T_q)
        query_mask = query_mask.unsqueeze(-1).repeat(1, 1, keys.shape[1])  # (h*N, T_q, T_k)
        matmul_output_qm = matmul_output_sm * query_mask

        # Dropout
        matmul_output_dropout = self.dropout(matmul_output_qm)

        # Weighted Sum
        output_ws = torch.bmm(matmul_output_dropout, V_)  # ( h*N, T_q, C/h)

        # Restore Shape
        output = torch.cat(torch.split(output_ws, output_ws.shape[0] // self.num_heads, dim=0), dim=2)  # (N, T_q, C)

        # Residual Connection
        output_res = output + queries

        return output_res


# Add diffusion to the list of safe globals so that safe loading accepts it.
torch.serialization.add_safe_globals([diffusion])
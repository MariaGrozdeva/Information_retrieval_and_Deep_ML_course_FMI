#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2025/2026
#############################################################################
###
### Машинен превод чрез генеративен езиков модел
###
#############################################################################

import torch
from parameters import *


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, seq_len=10000):
        super().__init__()

        pe = torch.zeros(1, seq_len, d_model)

        position = torch.arange(seq_len).unsqueeze(0).unsqueeze(2)
        div_term = (10000.0**(torch.arange(0, d_model, 2) /
                              d_model)).unsqueeze(0).unsqueeze(0)
        pe[0, :, 0::2] = torch.sin(position / div_term)
        pe[0, :, 1::2] = torch.cos(position / div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[
            1], :]  # x.shape = (batch_size, seq_len, embedding_dim)
        return x


class MultiHeadAttention(torch.nn.Module):

    def __init__(self, n_heads, d_model, d_keys, d_values):
        super().__init__()

        self.n_heads, self.d_keys, self.d_values = n_heads, d_keys, d_values
        self.scale = 1 / (d_keys**0.5)

        self.Wq_net = torch.nn.Linear(d_model, n_heads * d_keys)
        self.Wk_net = torch.nn.Linear(d_model, n_heads * d_keys)
        self.Wv_net = torch.nn.Linear(d_model, n_heads * d_values)
        self.Wo_net = torch.nn.Linear(n_heads * d_values, d_model)

    def forward(self, q_in, kv_in=None, attn_mask=None):

        if kv_in is None:  # none for self attention
            kv_in = q_in

        n_heads, d_keys, d_values = self.n_heads, self.d_keys, self.d_values

        batch_size_1, q_len, _ = q_in.shape
        batch_size_2, kv_len, _ = kv_in.shape
        assert batch_size_1 == batch_size_2

        head_q = self.Wq_net(q_in)
        head_k = self.Wk_net(kv_in)
        head_v = self.Wv_net(kv_in)
        # head_q.shape = (batch_size, q_len, n_heads * d_keys)
        # head_k.shape = (batch_size, kv_len, n_heads * d_keys)
        # head_v.shape = (batch_size, kv_len, n_heads * d_values)

        q = head_q.view(batch_size_1, q_len, n_heads, d_keys).transpose(1, 2)
        k = head_k.view(batch_size_1, kv_len, n_heads,
                        d_keys).permute(0, 2, 3, 1)
        v = head_v.view(batch_size_1, kv_len, n_heads,
                        d_values).transpose(1, 2)
        # q.shape = (batch_size, n_heads, q_len, d_keys)
        # k.shape = (batch_size, n_heads, d_keys, kv_len)
        # v.shape = (batch_size, n_heads, kv_len, d_values)

        attn_score = torch.matmul(q, k)
        # attn_score.shape = (batch_size, n_heads, q_len, kv_len)

        attn_score = attn_score * self.scale

        if attn_mask is not None:
            attn_score = attn_score.masked_fill(attn_mask, -float('inf'))

        attn_prob = torch.nn.functional.softmax(attn_score, dim=3)
        # attn_prob.shape = (batch_size, n_heads, q_len, kv_len)
        attn_vec = torch.matmul(attn_prob, v)
        # attn_vec.shape = (batch_size, n_heads, q_len, d_values)

        attn_vec = attn_vec.transpose(1, 2).contiguous().view(
            batch_size_1, q_len, n_heads * d_values)
        # attn_vec.shape = (batch_size, q_len, n_heads * d_values)

        # linear projection
        attn_out = self.Wo_net(attn_vec)
        # attn_out = (batch_size, q_len, d_model)
        return attn_out


class DecoderLayer(torch.nn.Module):

    def __init__(self, n_heads, d_model, d_keys, d_values, d_ff, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(n_heads, d_model, d_keys,
                                                 d_values)
        self.dropout_1 = torch.nn.Dropout(dropout)
        self.norm_1 = torch.nn.LayerNorm(d_model)
        self.W1 = torch.nn.Linear(d_model, d_ff)
        self.W2 = torch.nn.Linear(d_ff, d_model)
        self.dropout_2 = torch.nn.Dropout(dropout)
        self.norm_2 = torch.nn.LayerNorm(d_model)

    def forward(self, x, self_attn_mask=None):
        residual_x = x
        x = self.self_attention(x, attn_mask=self_attn_mask)
        x = self.dropout_1(x)
        x = self.norm_1(x + residual_x)

        residual_x = x
        x = self.W2(torch.nn.functional.relu(self.W1(x)))
        x = self.dropout_2(x)
        x = self.norm_2(x + residual_x)
        return x


class Decoder(torch.nn.Module):

    def __init__(self, n_layers, n_heads, d_model, d_keys, d_values, d_ff,
                 dropout):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            DecoderLayer(n_heads, d_model, d_keys, d_values, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, self_attn_mask=None):
        for layer in self.layers:
            x = layer(x, self_attn_mask)
        return x


class LanguageModel(torch.nn.Module):

    def __init__(self, n_layers, n_heads, d_model, d_keys, d_values, d_ff,
                 seq_len, dropout, startToken, endToken, unkToken, padToken,
                 word2ind):
        super().__init__()

        self.word2ind = word2ind
        self.ind2Word = {v: k for k, v in word2ind.items()}

        self.start_token = startToken
        self.start_tokenIdx = word2ind[startToken]
        self.end_token_Idx = word2ind[endToken]
        self.unk_token_Idx = word2ind[unkToken]
        self.pad_token_Idx = word2ind[padToken]

        self.embed = torch.nn.Embedding(len(word2ind), d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.seq_len = seq_len

        self.decoder = Decoder(n_layers, n_heads, d_model, d_keys, d_values,
                               d_ff, dropout)

        self.projection = torch.nn.Linear(d_model, len(word2ind))

        self.loss_function = torch.nn.functional.cross_entropy

    def preparePaddedBatch(self, source, word2ind, pad_idx, unk_idx):
        device = next(self.parameters()).device
        m = max(len(s) for s in source)
        sents = [[word2ind.get(w, unk_idx) for w in s] for s in source]
        sents_padded = [s + (m - len(s)) * [pad_idx] for s in sents]
        return torch.tensor(sents_padded, dtype=torch.long, device=device)

    def save(self, fileName):
        torch.save(self.state_dict(), fileName)

    def load(self, fileName):
        self.load_state_dict(
            torch.load(fileName, map_location=next(self.parameters()).device))

    def makeMask(self, target):
        pad_mask = (target == self.pad_token_Idx).unsqueeze(1).unsqueeze(
            2)  # (batch_size, 1, 1, kv_len)

        T = target.shape[1]
        future_mask = torch.triu(
            torch.ones((T, T), device=target.device, dtype=torch.bool),
            diagonal=1)  # every value above the main diagonal is T
        future_mask = future_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

        return pad_mask | future_mask  # True = set to -inf in the attention

    def forward(self, source):
        source_padded = self.preparePaddedBatch(source, self.word2ind,
                                                self.pad_token_Idx,
                                                self.unk_token_Idx)
        mask = self.makeMask(source_padded[:, :-1])

        input = self.dropout(
            self.pos_encoding(self.embed(source_padded[:, :-1])))
        output = self.decoder(input, mask)

        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)

        logits = self.projection(output)
        target_flat = source_padded[:, 1:].contiguous().view(-1)

        loss = self.loss_function(logits,
                                  target_flat,
                                  ignore_index=self.pad_token_Idx)
        return loss

    def generate(self, prefix, limit=1000):
        device = next(self.parameters()).device

        with torch.no_grad():
            for _ in range(limit):
                ids = [
                    self.word2ind.get(w, self.unk_token_Idx) for w in prefix
                ]
                x = torch.tensor(ids, dtype=torch.long,
                                 device=device).unsqueeze(0)  # (1, T)

                mask = self.makeMask(x)  # (1, 1, T, T)

                h = self.dropout(self.pos_encoding(
                    self.embed(x)))  # (1, T, d_model)
                h = self.decoder(h, mask)  # (1, T, d_model)
                logits = self.projection(h)  # (1, T, vocab)

                next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                next_tok = self.ind2Word[next_id]
                prefix.append(next_tok)

                if next_id == self.end_token_Idx:
                    break

        return prefix

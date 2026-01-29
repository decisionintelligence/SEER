import torch
import torch.nn as nn

from ts_benchmark.baselines.seer.layers.Embed import (
    AugmentedEmbedding
)
from ts_benchmark.baselines.seer.layers.RevIN import RevIN
from ts_benchmark.baselines.seer.utils.masked_attention import (
    Encoder,
    EncoderLayer,
    FullAttention,
    AttentionLayer,
)
from ..layers.Transformer_EncDec import Encoder, EncoderLayer


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class SEERModel(nn.Module):
    def __init__(self, config):
        super(SEERModel, self).__init__()
        self.n_vars = config.enc_in
        self.revin = RevIN(self.n_vars)
        self.patch_len = config.patch_len
        self.stride = config.stride
        self.n_heads = config.n_heads
        self.config = config
        padding = self.stride
        self.patch_num = int((config.seq_len - self.patch_len) / self.stride + 2)
        self.augmented_embedding = AugmentedEmbedding(
            self.config,
            config.d_model,
            self.patch_len,
            self.stride,
            padding,
            config.dropout,
        )
        self.score_threshold = config.score_threshold
        if self.score_threshold != 0:
            self.casual_encoder = Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            FullAttention(
                                True,
                                config.factor,
                                attention_dropout=config.dropout,
                                output_attention=config.output_attention,
                            ),
                            config.d_model,
                            config.n_heads,
                        ),
                        config.d_model,
                        config.d_ff,
                        dropout=config.dropout,
                        activation=config.activation,
                    )
                    for l in range(1)
                ],
                norm_layer=torch.nn.LayerNorm(config.d_model),
            )

        self.full_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            config.factor,
                            attention_dropout=config.dropout,
                            output_attention=config.output_attention,
                        ),
                        config.d_model,
                        config.n_heads,
                    ),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for l in range(config.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model),
        )

        self.head_nf = int(
            config.d_model
            * config.scaling
            * int((config.seq_len - self.patch_len) / self.stride + 2 + 1)
        )

        if self.config.scaling != 1:
            self.linear_projection = nn.Linear(
                config.d_model, int(config.d_model * config.scaling), bias=False
            )

        self.head = FlattenHead(
            config.enc_in,
            self.head_nf,
            config.pred_len,
            head_dropout=config.head_dropout,
        )

        if self.score_threshold != 0:
            self.linear_filter = nn.Sequential(
                nn.Linear(config.d_model, 1), nn.Sigmoid()
            )

    def replaced_embedding(self, enc_out, score):
        mask_zero = score != 0
        inv = torch.zeros_like(score)
        inv[mask_zero] = (1 / score[mask_zero]).detach()
        identity = score * inv
        cond = (score > self.score_threshold).float()
        raw_mask = identity * cond
        raw_mask[:, 0] = 1

        series_token = enc_out[:, 0:1, :].expand(-1, self.patch_num + 1, -1)
        series_token = series_token * (1 - raw_mask.unsqueeze(-1))
        enc_out = enc_out * raw_mask.unsqueeze(-1)
        enc_out = enc_out + series_token
        return enc_out

    def forward(self, input, input_mark):
        # [batch_size, seq_len, n_vars]
        input = self.revin(input, "norm")
        # [batch_size, n_vars, seq_len]
        x_enc = input.permute(0, 2, 1)
        # [batch_size * n_vars, n_patch+1, dim]
        enc_out, l_importance = self.augmented_embedding(x_enc, input_mark)

        if self.score_threshold != 0:
            # [batch_size * n_vars, n_patch+1]
            score = self.linear_filter(enc_out).squeeze(-1)
            # [batch_size * n_vars, n_patch+1, dim]
            enc_out = self.replaced_embedding(enc_out, score)
            # [batch_size * n_vars, n_patch+1, dim]
            enc_out, attns = self.casual_encoder(enc_out)
            # [batch_size * n_vars, n_patch+1, dim]
            enc_out, attns = self.full_encoder(enc_out)

        else:
            # [batch_size * n_vars, n_patch+1, dim]
            enc_out, attns = self.full_encoder(enc_out)
        # [batch_size, n_vars, n_patch+1, dim]
        enc_out = enc_out.reshape(-1, self.n_vars, enc_out.shape[-2], enc_out.shape[-1])

        if self.config.scaling != 1:
            enc_out = self.linear_projection(enc_out)
        # [batch_size, n_vars, horizon]
        dec_out = self.head(enc_out)
        # [batch_size, horizon, n_vars]
        dec_out = dec_out.permute(0, 2, 1)
        output = self.revin(dec_out, "denorm")
        return output, l_importance

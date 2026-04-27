import math

import torch
import torch.nn as nn
import torch.nn.functional as F



class RNADrugCrossAttentionPredictor(nn.Module):
    def __init__(self, in_dim=128, emb_dim=1024, num_tokens=16, token_dim=64):
        super().__init__()

        assert num_tokens * token_dim == emb_dim

        # -------- 128 → 1024 expansion --------
        self.rna_expand = nn.Linear(in_dim, emb_dim)
        self.drug_expand = nn.Linear(in_dim, emb_dim)

        self.num_tokens = num_tokens
        self.token_dim = token_dim

        # -------- Projection --------
        self.rna_proj = nn.Linear(emb_dim, emb_dim)
        self.drug_proj = nn.Linear(emb_dim, emb_dim)

        # -------- Cross Attention --------
        self.attn_rna_to_drug = nn.MultiheadAttention(
            embed_dim=token_dim, num_heads=4, batch_first=True
        )
        self.attn_drug_to_rna = nn.MultiheadAttention(
            embed_dim=token_dim, num_heads=4, batch_first=True
        )

        # -------- Fusion & Prediction --------
        self.fc = nn.Sequential(
            nn.Linear(2 * token_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, H_rna, H_drug, edge_index):

        rna_idx = edge_index[0]
        drug_idx = edge_index[1]

        # ---- 取出边对应的 embedding ----
        rna_emb = H_rna[rna_idx]  # [E, 128]
        drug_emb = H_drug[drug_idx]  # [E, 128]

        # ---- 128 → 1024 ----
        rna_emb = self.rna_expand(rna_emb)  # [E, 1024]
        drug_emb = self.drug_expand(drug_emb)  # [E, 1024]

        # ---- 投影并 reshape 为 [E, 16, 64] ----
        rna_tokens = self.rna_proj(rna_emb).view(
            -1, self.num_tokens, self.token_dim
        )
        drug_tokens = self.drug_proj(drug_emb).view(
            -1, self.num_tokens, self.token_dim
        )

        # ---- Cross Attention ----
        # RNA queries Drug
        rna_attn, _ = self.attn_rna_to_drug(
            query=rna_tokens,
            key=drug_tokens,
            value=drug_tokens
        )

        # Drug queries RNA
        drug_attn, _ = self.attn_drug_to_rna(
            query=drug_tokens,
            key=rna_tokens,
            value=rna_tokens
        )

        # ---- Pooling ----
        rna_vec = rna_attn.mean(dim=1)  # [E, 64]
        drug_vec = drug_attn.mean(dim=1)  # [E, 64]

        # ---- Fusion ----
        pair_repr = torch.cat([rna_vec, drug_vec], dim=-1)  # [E, 128]

        # ---- Prediction ----
        score = self.fc(pair_repr).squeeze(-1)  # [E]

        return score

class SelfAttention(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)

    def forward(self, x):

        Q = self.W_q(x)   # [N, d]
        K = self.W_k(x)   # [N, d]
        V = self.W_v(x)   # [N, d]

        # Attention scores
        scores = torch.matmul(Q, K.T) / math.sqrt(self.dim)  # [N, N]

        # Attention weights
        attn = F.softmax(scores, dim=-1)  # row-normalized

        # Weighted sum
        z = torch.matmul(attn, V)  # [N, d]

        return z, attn

class PairMLP(nn.Module):
    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.act(self.fc1(x))
        # h = self.dropout(h)
        h = h + self.act(self.fc2(h))   # 残差
        # h = self.act(self.fc2(h))
        h = self.dropout(h)
        return self.fc3(h)

class RNA_Drug_Predictor_Matrix(nn.Module):
    def __init__(self,
                 drug_dim=736,
                 mirna_dim_ex=938,
                 mirna_dim_seq=84,
                 hidden_dim=128,
                 cnn_channels=64,
                 kernel_sizes=[3,5,7],
                 dropout=0.3,
                 channels=32,
                 conv_layers=1,
                 use_attention=True):
        """
        conv_layers: RNA 和 Drug 卷积层数统一参数
        cnn_channels: 每层卷积通道数
        kernel_sizes: 每层卷积使用的卷积核列表
        """
        super().__init__()
        self.use_attention = use_attention  # 是否启用双向注意力机制

        # 可学习的融合权重
        self.fuse_weight = nn.Parameter(torch.tensor(0.5))  # 初始值0.5

        # ----- Drug 特征卷积 -----
        self.drug_convs1 = nn.ModuleList()
        in_channels = channels
        for i in range(conv_layers):
            conv_block = nn.ModuleList([
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=cnn_channels,
                    kernel_size=k,
                    padding=k // 2
                )
                for k in kernel_sizes
            ])
            self.drug_convs1.append(conv_block)
            # 对于下一层卷积，输入通道等于上一层输出的总通道数
            in_channels = cnn_channels * len(kernel_sizes)

        self.fc_drug = nn.Sequential(
            nn.Linear(cnn_channels * len(kernel_sizes), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # ----- RNA 表达特征卷积 -----
        self.rna_ex_convs1 = nn.ModuleList()
        in_channels = channels
        for i in range(conv_layers):
            conv_block = nn.ModuleList([
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=cnn_channels,
                    kernel_size=k,
                    padding=k // 2
                )
                for k in kernel_sizes
            ])
            self.rna_ex_convs1.append(conv_block)
            in_channels = cnn_channels * len(kernel_sizes)

        self.fc_rna_ex = nn.Sequential(
            nn.Linear(cnn_channels * len(kernel_sizes), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ----- RNA 序列特征卷积 -----
        self.rna_seq_convs1 = nn.ModuleList()
        in_channels = channels
        for i in range(conv_layers):
            conv_block = nn.ModuleList([
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=cnn_channels,
                    kernel_size=k,
                    padding=k // 2
                )
                for k in kernel_sizes
            ])
            self.rna_seq_convs1.append(conv_block)
            in_channels = cnn_channels * len(kernel_sizes)

        self.fc_rna_seq = nn.Sequential(
            nn.Linear(cnn_channels * len(kernel_sizes), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.predictor_ex = RNADrugCrossAttentionPredictor(in_dim=hidden_dim)
        self.predictor_seq = RNADrugCrossAttentionPredictor(in_dim=hidden_dim)

        self.mlp = PairMLP(hidden_dim=hidden_dim)

        self.drug_fusion = nn.Linear(256, 128)

        # ----- 注意力机制模块（双向 IIP） -----
        if use_attention:

            self.self_att_ex = SelfAttention(dim=hidden_dim)
            self.self_att_seq = SelfAttention(dim=hidden_dim)
            self.self_att_drug = SelfAttention(dim=hidden_dim)

        self.drug_fuse_mlp = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.RNA_fuse_mlp = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        # ----- 预测层 -----
        self.fc_pair = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # 输出 logit
        )


    def _conv_forward(self, x, conv_blocks):

        in_channels = conv_blocks[0][0].in_channels

        x = x.unsqueeze(1)  # [batch, 1, feature_dim]
        x = x.repeat(1, in_channels, 1)  # [batch, channels, feature_dim]

        # ----- CNN 提取特征 -----
        for conv_block in conv_blocks:
            conv_outs = []
            for conv in conv_block:
                y = F.relu(conv(x))  # [batch, cnn_channels, feature_dim]
                y = F.adaptive_max_pool1d(y, 1).squeeze(2)  # [batch, cnn_channels]
                conv_outs.append(y)
            x = torch.cat(conv_outs, dim=1).unsqueeze(2)  # [batch, cnn_channels*len(kernel_sizes), 1]
        out = x.squeeze(2)  # [batch, cnn_channels*len(kernel_sizes)]
        return out

    def _predict_pair(self, rna_emb, drug_emb, edge_index):
        """
        根据边索引生成预测 logits
        """
        rna_edges = rna_emb[edge_index[0]]      # [num_edges, hidden_dim]
        drug_edges = drug_emb[edge_index[1]]    # [num_edges, hidden_dim]
        pair_feat = torch.cat([rna_edges, drug_edges], dim=1)
        logits = self.mlp(pair_feat).squeeze(1)
        return logits

    def forward(self, rna_ex, rna_seq, drug_feat, edge_index=None):
        #----- RNA embedding -----
        rna_ex_emb1 = self._conv_forward(rna_ex, self.rna_ex_convs1)
        rna_ex_emb = self.fc_rna_ex(rna_ex_emb1)

        rna_seq_emb1 = self._conv_forward(rna_seq, self.rna_seq_convs1)
        rna_seq_emb = self.fc_rna_seq(rna_seq_emb1)

        # ----- Drug embedding -----
        drug_emb1 = self._conv_forward(drug_feat, self.drug_convs1)
        drug_emb = self.fc_drug(drug_emb1)

        if self.use_attention:

            _, att1 = self.self_att_ex(rna_ex_emb)
            _, att2 = self.self_att_seq(rna_seq_emb)
            _, att3 = self.self_att_drug(drug_emb)

            # att1 = rna_ex_emb @ rna_ex_emb.T
            # att2 = rna_seq_emb @ rna_seq_emb.T
            # att3 = drug_emb @ drug_emb.T

            att_3d = torch.stack([att1, att2], dim=0)


            scores1 = self.predictor_ex(
                H_rna=rna_ex_emb,  # [1927, 128]
                H_drug=drug_emb,  # [216, 128]
                edge_index=edge_index  # [2, E]
            )
            scores2 = self.predictor_seq(
                H_rna=rna_seq_emb,  # [1927, 128]
                H_drug=drug_emb,  # [216, 128]
                edge_index=edge_index  # [2, E]
            )

            # scores1 = self._predict_pair(rna_ex_emb,drug_emb,edge_index)
            # scores2 = self._predict_pair(rna_seq_emb, drug_emb, edge_index)

            scores = scores1 + scores2

            return scores, rna_seq_emb, rna_ex_emb, drug_emb, att_3d, att3

    # only seq
    # def forward(self, rna_seq, drug_feat, edge_index=None):
    #
    #     rna_seq_emb1 = self._conv_forward(rna_seq, self.rna_seq_convs1)
    #     rna_seq_emb = self.fc_rna_seq(rna_seq_emb1)
    #
    #     # ----- Drug embedding -----
    #     drug_emb1 = self._conv_forward(drug_feat, self.drug_convs1)
    #     drug_emb = self.fc_drug(drug_emb1)
    #
    #     if self.use_attention:
    #
    #         _, att1 = self.self_att_seq(rna_seq_emb)
    #         _, att2 = self.self_att_drug(drug_emb)
    #
    #         # att1 = rna_ex_emb @ rna_ex_emb.T
    #         # att2 = rna_seq_emb @ rna_seq_emb.T
    #         # att3 = drug_emb @ drug_emb.T
    #
    #         scores = self.predictor_seq(
    #             H_rna=rna_seq_emb,  # [1927, 128]
    #             H_drug=drug_emb,  # [216, 128]
    #             edge_index=edge_index  # [2, E]
    #         )
    #
    #         return scores, rna_seq_emb, drug_emb, att1, att2

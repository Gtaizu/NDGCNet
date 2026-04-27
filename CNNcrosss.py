import torch
import torch.nn as nn
import torch.nn.functional as F


class BilinearAttention(nn.Module):

    def __init__(self, d, K):

        super(BilinearAttention, self).__init__()
        self.U = nn.Parameter(torch.randn(d, K))  # antigen projection
        self.V = nn.Parameter(torch.randn(K, d))  # antibody projection
        self.q = nn.Parameter(torch.randn(K))  # attention vector
        self.dropout = nn.Dropout(p=0.2)
        self.layernorm_ag = nn.LayerNorm(K)
        self.layernorm_ab = nn.LayerNorm(K)

    def forward(self, H_ag, H_ab):

        M, d1 = H_ag.shape
        N, d2 = H_ab.shape
        assert d1 == self.U.shape[0] and d2 == self.V.shape[1], "Feature dimension mismatch"

        # 1. Antigen projection and attention modulation
        H_ag_proj = torch.tanh(H_ag @ self.U)
        H_ag_proj = self.dropout(H_ag_proj)
        H_ag_proj = self.layernorm_ag(H_ag_proj)
        q_matrix = self.q.unsqueeze(0).repeat(M, 1)  # (M, K)
        H_ag_att = H_ag_proj * q_matrix  # Hadamard product (M, K)

        # 2. Antibody projection
        H_ab_proj = torch.tanh(self.V @ H_ab.T)
        H_ab_proj = self.dropout(H_ab_proj)
        H_ab_proj = self.layernorm_ab(H_ab_proj.T).T

        # 3. Bilinear interaction
        I = H_ag_att @ H_ab_proj  # (M, N)

        I = torch.softmax(I, dim=1)

        return I

class CrossAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_k)
        self.key_proj = nn.Linear(d_model, d_k)
        self.value_proj = nn.Linear(d_model, d_k)

        self.scale = d_k ** 0.5

    def forward(self, Q_in, K_in):
        # Q_in: [num_drug, d_model]
        # K_in: [num_rna,  d_model]

        Q = self.query_proj(Q_in)  # [216, d_k]
        K = self.key_proj(K_in)  # [1927, d_k]
        V = self.value_proj(K_in)  # [1927, d_k]

        # ---- attention scores: matrix multiplication ----
        # scores: [216, 1927]
        scores = Q @ K.T / self.scale

        attn = torch.softmax(scores, dim=-1)  # [216, 1927]

        # ---- weighted sum ----
        out = attn @ V  # [216, d_k]

        return out

class FeatureCrossAttention(nn.Module):
    def __init__(self, d_model, d_k, n_slots=216, slot_mode='repeat'):
        """
        slot_mode:
          - 'repeat' : 每个 RNA 直接重复自身到 n_slots（简单、无额外参数）
          - 'pos_embed' : 使用可学习的 slot-wise transform: slot_proj(linear) + rna.unsqueeze(1)
          - 'mlp' : 使用一个小 MLP 将 rna -> [n_slots, d_model] （可学习，每个 RNA 有不同 slots）
        """
        super().__init__()
        self.n_slots = n_slots
        self.d_k = d_k
        self.slot_mode = slot_mode

        # Projections
        self.q_proj = nn.Linear(d_model, d_k)
        self.k_proj = nn.Linear(d_model, d_k)
        self.v_proj = nn.Linear(d_model, d_k)

        # optional slot projection when slot_mode != 'repeat'
        if slot_mode == 'pos_embed':
            # learnable slot offsets (shared across RNAs)
            self.slot_offsets = nn.Parameter(torch.randn(n_slots, d_model) * 0.01)
        elif slot_mode == 'mlp':
            # map rna_emb -> (n_slots * d_model) then reshape
            self.slot_mlp = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, n_slots * d_model)
            )

        self.scale = d_k ** 0.5

    def forward(self, rna_emb, drug_emb, reduce_mode='mean'):
        """
        rna_emb:  [N_rna, d_model]  (e.g. [1927,128])
        drug_emb: [N_drug, d_model]  (e.g. [216,128])   -- note: we require N_drug == n_slots typically
        reduce_mode: how to reduce per-RNA [n_slots, d_k] -> [d_k], choices: 'mean','max','attn','none'
        returns:
           out_full: [N_rna, n_slots, d_k]  (if reduce_mode=='none' or for inspection)
           out_reduced: [N_rna, d_k] (if reduce_mode != 'none')
        """
        N_rna = rna_emb.size(0)
        N_drug = drug_emb.size(0)
        assert N_drug == self.n_slots, "For your described pipeline N_drug must equal n_slots (216)."

        # --------- build Q_slots: [N_rna, n_slots, d_model] ----------
        if self.slot_mode == 'repeat':
            # simplest: repeat RNA vector across slots
            Q_slots = rna_emb.unsqueeze(1).expand(-1, self.n_slots, -1).contiguous()
            # shape: [N_rna, n_slots, d_model]
        elif self.slot_mode == 'pos_embed':
            # add learnable slot offsets (shared across RNAs)
            Q_slots = rna_emb.unsqueeze(1) + self.slot_offsets.unsqueeze(0)   # [N_rna, n_slots, d_model]
        else:  # 'mlp'
            x = self.slot_mlp(rna_emb)   # [N_rna, n_slots * d_model]
            Q_slots = x.view(N_rna, self.n_slots, -1)  # [N_rna, n_slots, d_model]

        # --------- project Q, K, V ----------
        # Q: [N_rna, n_slots, d_k]
        Q = self.q_proj(Q_slots)   # vectorized
        # K: [n_slots, d_k]
        K = self.k_proj(drug_emb)
        # V: [n_slots, d_k]
        V = self.v_proj(drug_emb)

        # --------- compute batched scores ---------
        # We want scores: [N_rna, n_slots, n_slots] where scores[i] = Q[i] @ K.T
        # use einsum for clarity and efficiency:
        # scores[n, q, k] = sum_d Q[n,q,d] * K[k,d]
        scores = torch.einsum('nqd,kd->nqk', Q, K) / self.scale   # [N_rna, n_slots, n_slots]

        attn = torch.softmax(scores, dim=-1)   # softmax over last dim -> [N_rna, n_slots, n_slots]

        # --------- weighted values: out_full [N_rna, n_slots, d_k] ----------
        # out[n, q, d] = sum_k attn[n,q,k] * V[k,d]
        out_full = torch.einsum('nqk,kd->nqd', attn, V)   # [N_rna, n_slots, d_k]

        if reduce_mode == 'none':
            return out_full   # [N_rna, n_slots, d_k]

        # --------- reduce per-RNA n_slots -> single vector ----------
        if reduce_mode == 'mean':
            out_reduced = out_full.mean(dim=1)   # [N_rna, d_k]
        elif reduce_mode == 'max':
            out_reduced, _ = out_full.max(dim=1)
        elif reduce_mode == 'attn':
            # 使用 Q 的聚合分数再做一个 slot->scalar 的注意力（可学习）
            # 简单实现：用 Q @ w -> logits, softmax over slots, 然后加权 out_full
            # 这里示例用 Q.mean(d_k) 的简易权重（可改成线性+非线性）
            slot_logits = Q.mean(dim=-1)          # [N_rna, n_slots]
            slot_w = torch.softmax(slot_logits, dim=-1).unsqueeze(-1)  # [N_rna, n_slots, 1]
            out_reduced = (out_full * slot_w).sum(dim=1)   # [N_rna, d_k]
        else:
            raise ValueError("unknown reduce_mode")

        return out_reduced   # [N_rna, d_k]

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

        # self.drug_convs2 = nn.ModuleList()
        # in_channels = channels
        # for i in range(conv_layers):
        #     conv_block = nn.ModuleList([
        #         nn.Conv1d(
        #             in_channels=in_channels,
        #             out_channels=cnn_channels,
        #             kernel_size=k,
        #             padding=k // 2
        #         )
        #         for k in kernel_sizes
        #     ])
        #     self.drug_convs2.append(conv_block)
        #     # 对于下一层卷积，输入通道等于上一层输出的总通道数
        #     in_channels = cnn_channels * len(kernel_sizes)

        self.fc_drug = nn.Sequential(
            nn.Linear(cnn_channels * len(kernel_sizes), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # self.fc_drug1 = nn.Sequential(
        #     nn.Linear(256, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout)
        # )

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

        # self.rna_ex_convs2 = nn.ModuleList()
        # in_channels = channels
        # for i in range(conv_layers):
        #     conv_block = nn.ModuleList([
        #         nn.Conv1d(
        #             in_channels=in_channels,
        #             out_channels=cnn_channels,
        #             kernel_size=k,
        #             padding=k // 2
        #         )
        #         for k in kernel_sizes
        #     ])
        #     self.rna_ex_convs2.append(conv_block)
        #     in_channels = cnn_channels * len(kernel_sizes)

        self.fc_rna_ex = nn.Sequential(
            nn.Linear(cnn_channels * len(kernel_sizes), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # self.fc_rna_ex1 = nn.Sequential(
        #     nn.Linear(256, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout)
        # )

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

        # self.rna_seq_convs2 = nn.ModuleList()
        # in_channels = channels
        # for i in range(conv_layers):
        #     conv_block = nn.ModuleList([
        #         nn.Conv1d(
        #             in_channels=in_channels,
        #             out_channels=cnn_channels,
        #             kernel_size=k,
        #             padding=k // 2
        #         )
        #         for k in kernel_sizes
        #     ])
        #     self.rna_seq_convs2.append(conv_block)
        #     in_channels = cnn_channels * len(kernel_sizes)

        self.fc_rna_seq = nn.Sequential(
            nn.Linear(cnn_channels * len(kernel_sizes), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # self.fc_rna_seq1 = nn.Sequential(
        #     nn.Linear(256, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout)
        # )

        # self.att_ex_convs = nn.ModuleList([
        #     nn.Conv1d(1, cnn_channels, kernel_size=k, padding=k // 2)
        #     for k in kernel_sizes
        # ])
        # self.att_seq_convs = nn.ModuleList([
        #     nn.Conv1d(1, cnn_channels, kernel_size=k, padding=k // 2)
        #     for k in kernel_sizes
        # ])
        # self.att_drug_convs = nn.ModuleList([
        #     nn.Conv1d(1, cnn_channels, kernel_size=k, padding=k // 2)
        #     for k in kernel_sizes
        # ])

        # self.pre_norm_ex = nn.LayerNorm(mirna_dim_ex)
        # self.pre_norm_seq = nn.LayerNorm(mirna_dim_seq)
        # self.pre_norm_drug = nn.LayerNorm(drug_dim)

        # ----- 注意力机制模块（双向 IIP） -----
        if use_attention:
            # self.att_ex = BilinearAttention(d=hidden_dim, K=512)
            # self.att_seq = BilinearAttention(d=hidden_dim, K=512)
            # RNA → Drug cross attention
            self.cross_att_ex = CrossAttention(d_model=hidden_dim, d_k=hidden_dim)
            self.cross_att_seq = CrossAttention(d_model=hidden_dim, d_k=hidden_dim)

            self.cross_att_exdrug = CrossAttention(d_model=hidden_dim, d_k=hidden_dim)
            self.cross_att_seqdrug = CrossAttention(d_model=hidden_dim, d_k=hidden_dim)

            self.Fcross_att_exdrug = FeatureCrossAttention(d_model=hidden_dim, d_k=hidden_dim)
            self.Fcross_att_seqdrug = FeatureCrossAttention(d_model=hidden_dim, d_k=hidden_dim)

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

        self.mlp = PairMLP(hidden_dim=hidden_dim)

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
        # rna_ex_emb1 = self.fc_rna_ex(rna_ex_emb1)
        rna_ex_emb = self.fc_rna_ex(rna_ex_emb1)
        # rna_ex_emb2 = self._conv_forward(rna_ex, self.rna_ex_convs2)
        # rna_ex_emb2 = self.fc_rna_ex(rna_ex_emb2)
        #
        # rna_ex_emb = torch.cat([rna_ex_emb1, rna_ex_emb2], dim=1)  # [batch, hidden_dim*2]
        # # 压缩回 hidden_dim
        # rna_ex_emb = self.fc_rna_ex1(rna_ex_emb)

        rna_seq_emb1 = self._conv_forward(rna_seq, self.rna_seq_convs1)
        # rna_seq_emb1 = self.fc_rna_seq(rna_seq_emb1)
        rna_seq_emb = self.fc_rna_seq(rna_seq_emb1)
        # rna_seq_emb2 = self._conv_forward(rna_seq, self.rna_seq_convs2)
        # rna_seq_emb2 = self.fc_rna_seq(rna_seq_emb2)
        #
        # rna_seq_emb = torch.cat([rna_seq_emb1, rna_seq_emb2], dim=1)  # [batch, hidden_dim*2]
        # # 压缩回 hidden_dim
        # rna_seq_emb = self.fc_rna_seq1(rna_seq_emb)

        # rna_seq_emb = rna_seq
        # rna_ex_emb = rna_ex

        # ----- Drug embedding -----
        drug_emb1 = self._conv_forward(drug_feat, self.drug_convs1)
        # drug_emb1 = self.fc_drug(drug_emb1)
        drug_emb = self.fc_drug(drug_emb1)
        # drug_emb2 = self._conv_forward(drug_feat, self.drug_convs2)
        # drug_emb2 = self.fc_drug(drug_emb2)
        #
        # # 拼接两个嵌入矩阵，得到新的药物嵌入
        # drug_emb = torch.cat([drug_emb1, drug_emb2], dim=1)  # [batch, hidden_dim*2]
        # drug_emb = self.fc_drug1(drug_emb)


        # ===== 新增：双向注意力机制（基于交互矩阵） =====
        if self.use_attention:
            # # ---- Attention: RNA_ex → Drug ----
            # drug_emb_ex = self.cross_att_ex(drug_emb, rna_ex_emb)
            #
            # # ---- Attention: RNA_seq → Drug ----
            # drug_emb_seq = self.cross_att_seq(drug_emb, rna_seq_emb)

            # rna_ex_emb = self.cross_att_exdrug(rna_ex_emb, drug_emb)
            # rna_seq_emb = self.cross_att_seqdrug(rna_seq_emb, drug_emb)

            rna_ex_emb = self.Fcross_att_exdrug(rna_ex_emb, drug_emb)
            rna_seq_emb = self.Fcross_att_seqdrug(rna_seq_emb, drug_emb)

            # drug_cat = torch.cat([drug_emb_ex, drug_emb_seq], dim=1)
            # drug_emb = self.drug_fuse_mlp(drug_cat)

            # drug_emb = drug_emb_ex

            RNA_cat = torch.cat([rna_ex_emb, rna_seq_emb], dim=1)
            RNA_emb = self.RNA_fuse_mlp(RNA_cat)

            # RNA_emb = rna_seq_emb

        # ===== 原始预测逻辑 =====
        if edge_index is not None:
            # 分别计算两类 RNA 的预测
            # pred_ex = self._predict_pair(rna_ex_emb, drug_emb, edge_index)
            # pred_seq = self._predict_pair(rna_seq_emb, drug_emb, edge_index)
            #
            # w = torch.sigmoid(self.fuse_weight)  # 范围 [0,1]
            # logits = w * pred_ex + (1 - w) * pred_seq

            pred = self._predict_pair(RNA_emb, drug_emb, edge_index)
            logits = pred
            # logits = pred_ex
            # logits = pred_seq

            return logits, rna_seq_emb, rna_ex_emb, drug_emb
        else:
            # 返回 embedding，便于消融实验
            return rna_ex_emb, rna_seq_emb, drug_emb
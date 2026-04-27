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

        self.lncrna_ex_convs = nn.ModuleList()
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
            self.lncrna_ex_convs.append(conv_block)
            in_channels = cnn_channels * len(kernel_sizes)

        self.fc_lncrna_ex = nn.Sequential(
            nn.Linear(cnn_channels * len(kernel_sizes), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.mirna_ex_convs = nn.ModuleList()
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
            self.mirna_ex_convs.append(conv_block)
            in_channels = cnn_channels * len(kernel_sizes)

        self.fc_mirna_ex = nn.Sequential(
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

        self.lncrna_seq_convs = nn.ModuleList()
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
            self.lncrna_seq_convs.append(conv_block)
            in_channels = cnn_channels * len(kernel_sizes)

        self.fc_lncrna_seq = nn.Sequential(
            nn.Linear(cnn_channels * len(kernel_sizes), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.mirna_seq_convs = nn.ModuleList()
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
            self.mirna_seq_convs.append(conv_block)
            in_channels = cnn_channels * len(kernel_sizes)

        self.fc_mirna_seq = nn.Sequential(
            nn.Linear(cnn_channels * len(kernel_sizes), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ----- 注意力机制模块（双向 IIP） -----
        if use_attention:
            self.att_ex = BilinearAttention(d=hidden_dim, K=512)
            self.att_seq = BilinearAttention(d=hidden_dim, K=512)

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
        # ----- RNA embedding -----
        rna_ex_emb1 = self._conv_forward(rna_ex, self.rna_ex_convs1)
        rna_ex_emb = self.fc_rna_ex(rna_ex_emb1)

        rna_seq_emb1 = self._conv_forward(rna_seq, self.rna_seq_convs1)
        rna_seq_emb = self.fc_rna_seq(rna_seq_emb1)

        # ----- Drug embedding -----
        drug_emb1 = self._conv_forward(drug_feat, self.drug_convs1)
        drug_emb = self.fc_drug(drug_emb1)

        lnc_idx = slice(0, 1322)
        mirna_idx = slice(1322, 1927)

        lnc_ex = rna_ex[lnc_idx]
        mirna_ex = rna_ex[mirna_idx]

        lnc_seq = rna_seq[lnc_idx]
        mirna_seq = rna_seq[mirna_idx]

        lnc_ex_emb = self._conv_forward(lnc_ex, self.lncrna_ex_convs)
        lnc_ex_emb = self.fc_mirna_ex(lnc_ex_emb)

        mi_ex_emb = self._conv_forward(mirna_ex, self.mirna_ex_convs)
        mi_ex_emb = self.fc_mirna_ex(mi_ex_emb)

        lnc_seq_emb = self._conv_forward(lnc_seq, self.lncrna_seq_convs)
        lnc_seq_emb = self.fc_lncrna_seq(lnc_seq_emb)

        mi_seq_emb = self._conv_forward(mirna_seq, self.mirna_seq_convs)
        mi_seq_emb = self.fc_mirna_seq(mi_seq_emb)

        ex_emb = torch.cat([lnc_ex_emb, mi_ex_emb], dim=0)
        seq_emb = torch.cat([lnc_seq_emb, mi_seq_emb], dim=0)

        # ex_emb = rna_ex
        # seq_emb = rna_seq

        # ===== 新增：双向注意力机制（基于交互矩阵） =====
        if self.use_attention:
            att_ex = self.att_ex(rna_ex_emb, drug_emb)
            att_seq = self.att_seq(rna_seq_emb, drug_emb)

            rna_ex_emb = att_ex @ drug_emb
            rna_seq_emb = att_seq @ drug_emb


        # ===== 原始预测逻辑 =====
        if edge_index is not None:
            # 分别计算两类 RNA 的预测
            pred_ex = self._predict_pair(rna_ex_emb, drug_emb, edge_index)
            pred_seq = self._predict_pair(rna_seq_emb, drug_emb, edge_index)

            pred_ex_p = self._predict_pair(ex_emb, drug_emb, edge_index)
            pred_seq_p = self._predict_pair(seq_emb, drug_emb, edge_index)

            # #
            w = torch.sigmoid(self.fuse_weight)  # 范围 [0,1]
            logits = w * pred_ex + (1 - w) * pred_seq

            logits1 = 0.5 * pred_ex_p + (1 - 0.5) * pred_seq_p

            g = torch.sigmoid(logits1)

            logits = logits + g * logits1

            # logits = pred_ex
            # logits = pred_seq

            return logits, rna_seq_emb, rna_ex_emb, drug_emb, ex_emb, seq_emb
        else:
            # 返回 embedding，便于消融实验
            return rna_ex_emb, rna_seq_emb, drug_emb
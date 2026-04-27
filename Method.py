import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    matthews_corrcoef,
    brier_score_loss,
    roc_curve,
    precision_recall_curve
)
import torch.nn.functional as F

def sample_negative_edges(mask, num_neg):
    """一次性负采样"""
    eligible = (~mask).nonzero(as_tuple=False)
    perm = torch.randperm(eligible.size(0), device=mask.device)
    selected = eligible[perm[:num_neg]].T
    return selected


def compute_metrics(y_true, y_score, method='f1'):

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    J = tpr - fpr
    best_thresh_roc = roc_thresholds[np.argmax(J)]

    # 使用 PR 曲线计算 F1-max 阈值
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    F1 = 2 * precision * recall / (precision + recall + 1e-8)
    # 注意 pr_thresholds 比 precision/recall少一个点，需要对齐
    best_thresh_pr = pr_thresholds[np.argmax(F1[:-1])]

    # 根据 method 选择阈值
    if method == 'f1':
        best_thresh = best_thresh_pr
    elif method == 'roc':
        best_thresh = best_thresh_roc
    else:
        # 默认使用 0.5
        best_thresh = 0.5

    y_pred = (y_score >= best_thresh).astype(int)

    auc = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    brier = brier_score_loss(y_true, y_score)

    return auc, aupr, acc, f1, pre, mcc, brier, best_thresh

def compute_metrics1(y_true, y_score):
    y_pred = (y_score >= 0.5).astype(int)
    auc = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    brier = brier_score_loss(y_true, y_score)
    return auc, aupr, acc, f1, pre, mcc, brier, 0.5

def compute_type_contrastive_loss(rna_seq_emb, num_lnc=1322, temperature=0.5):

    # 拆分为 lncRNA 和 miRNA
    lnc_emb = rna_seq_emb[:num_lnc]  # [1322, 128]
    mi_emb = rna_seq_emb[num_lnc:]   # [605, 128]
    # 归一化嵌入（计算余弦相似度）
    lnc_emb = F.normalize(lnc_emb, p=2, dim=1)
    mi_emb = F.normalize(mi_emb, p=2, dim=1)
    # 计算相似度矩阵 (lnc x miRNA)
    sim_matrix = torch.matmul(lnc_emb, mi_emb.T) / temperature  # [1322, 605]
    # 负样本：lncRNA 与 miRNA 的相似度越低越好
    lnc_to_mi_loss = -torch.log(1 - torch.sigmoid(sim_matrix)).mean()
    mi_to_lnc_loss = -torch.log(1 - torch.sigmoid(sim_matrix.T)).mean()
    # 正样本：同类之间应高相似度
    sim_lnc = torch.matmul(lnc_emb, lnc_emb.T) / temperature
    sim_mi = torch.matmul(mi_emb, mi_emb.T) / temperature
    same_type_loss = -torch.log(torch.sigmoid(sim_lnc)).mean() - torch.log(torch.sigmoid(sim_mi)).mean()
    # 综合损失
    contrastive_loss = same_type_loss + (lnc_to_mi_loss + mi_to_lnc_loss) / 2.0

    return contrastive_loss

def compute_contrastive_loss(embeddings, num_lnc=1322, temperature=0.5):

    N = embeddings.size(0)

    embeddings = F.normalize(embeddings, dim=-1)

    sim = torch.matmul(embeddings, embeddings.T) / temperature  # (N, N)

    labels = torch.zeros(N, dtype=torch.long, device=embeddings.device)
    labels[num_lnc:] = 1
    pos_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
    pos_mask.fill_diagonal_(0)

    log_denom = torch.logsumexp(sim, dim=1)  # (N,)

    masked_sim = sim + (pos_mask == 0).float() * -1e9
    log_numer = torch.logsumexp(masked_sim, dim=1)

    valid = pos_mask.sum(dim=1) > 0
    if valid.any():
        loss = -((log_numer - log_denom)[valid]).mean()
    else:
        loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    return loss

def compute_contrastive_loss_with_topk(
        embeddings,
        drug_emb,
        num_lnc=1322,
        num_mir=605,
        top_k=512,
        temperature=0.5
    ):

    N = embeddings.size(0)
    emb = F.normalize(embeddings, dim=-1)

    # 全相似度矩阵（用于同类型RNA）
    sim_full = emb @ emb.t() / temperature     # [1927, 1927]

    # 标签：lncRNA=0, miRNA=1
    labels = torch.zeros(N, dtype=torch.long, device=embeddings.device)
    labels[num_lnc:] = 1     # miRNA 的标签为 1

    # mask：同类型为正样本
    pos_mask_same = torch.eq(labels[:, None], labels[None, :]).float()
    pos_mask_same.fill_diagonal_(0)

    # -----------------------------
    # PART 2：跨类型 top-K 正样本（miRNA–lncRNA）
    # -----------------------------
    if drug_emb is not None:
        # 分离 lncRNA 和 miRNA 嵌入
        lnc_emb = F.normalize(embeddings[:num_lnc], dim=-1)    # [1322, 128]
        mir_emb = F.normalize(embeddings[num_lnc:], dim=-1)    # [605, 128]
        drug_emb = F.normalize(drug_emb, dim=-1)               # [216, 128]

        # drug-mediated sim
        M_D = mir_emb @ drug_emb.t()       # [605, 216]
        L_D = lnc_emb @ drug_emb.t()       # [1322, 216]
        sim_lm = M_D @ L_D.t()             # [605, 1322]

        # flatten 选 top-K
        sim_flat = sim_lm.reshape(-1)
        top_vals, top_idx = torch.topk(sim_flat, top_k)

        mir_idx = top_idx // sim_lm.size(1)        # miRNA 行
        lnc_idx = top_idx % sim_lm.size(1)         # lncRNA 列

        # 转换到 embeddings 的全局索引空间
        mir_global = mir_idx + num_lnc             # 因为 miRNA 在后面
        lnc_global = lnc_idx

        # 构造跨类型正样本 mask
        pos_mask_cross = torch.zeros_like(sim_full)
        pos_mask_cross[mir_global, lnc_global] = 1
        pos_mask_cross[lnc_global, mir_global] = 1   # 对称

    else:
        pos_mask_cross = torch.zeros_like(sim_full)

    # -----------------------------
    # 合并正样本
    # -----------------------------
    pos_mask = pos_mask_same + pos_mask_cross
    pos_mask = (pos_mask > 0).float()

    # -----------------------------
    # InfoNCE 计算：对每个样本
    # log(numerator) = 所有正样本的 logsumexp
    # log(denominator) = 所有样本的 logsumexp
    # -----------------------------
    # denominator
    log_denom = torch.logsumexp(sim_full, dim=1)  # [1927]

    # numerator：只保留正样本的 logsumexp
    masked_sim = sim_full + (pos_mask == 0).float() * -1e9
    log_numer = torch.logsumexp(masked_sim, dim=1)  # [1927]

    valid = pos_mask.sum(dim=1) > 0
    loss = -(log_numer[valid] - log_denom[valid]).mean()

    return loss

def compute_mes(score_matrix, edge_index, edge_label):

    pred_scores = score_matrix[
        edge_index[0], edge_index[1]
    ].float()  # [num]
    labels = edge_label.float()

    mes = torch.mean((labels - pred_scores) ** 2)

    return mes

def compute_mse(pred_scores: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
    """
    Compute Mean Squared Error (MSE) with type safety.
    """
    assert pred_scores.shape == labels.shape

    pred_scores = pred_scores.float()
    labels = labels.float()

    return torch.mean((labels - pred_scores) ** 2)

def kl_attention_gaussian(attn, gaussian, eps=1e-10):
    """
    attn: self-attention matrix [n, n] or [B, n, n]
    gaussian: Gaussian interaction kernel matrix, same shape
    """

    # ---- 1. 数值安全 ----
    attn = attn.clamp(min=eps)
    gaussian = gaussian.clamp(min=eps)

    # ---- 2. 行归一化 → 分布 ----
    attn = attn / attn.sum(dim=-1, keepdim=True)
    gaussian = gaussian / gaussian.sum(dim=-1, keepdim=True)

    # ---- 3. KL(G || A) ----
    kl = F.kl_div(
        torch.log(attn),     # input = log P
        gaussian,            # target = Q
        reduction='batchmean'
    )

    return kl

def compute_ce(score_matrix, edge_index, edge_label):
    pred_scores = score_matrix[
        edge_index[0], edge_index[1]
    ].float()  # [num_edges] in [0,1]

    labels = edge_label.float()  # 0/1

    loss = F.binary_cross_entropy(pred_scores, labels)

    return loss

def gaussian_interaction_profile_kernel(Y: torch.Tensor, axis: int = 0):

    if axis == 1:
        Y = Y.T  # drug profiles
    # Y shape: [N, D]
    Y = Y.float()

    # Squared L2 norm of each interaction profile
    norm_sq = torch.sum(Y ** 2, dim=1)  # [N]

    # gamma = 1 / mean(||y_i||^2)
    gamma = 1.0 / torch.mean(norm_sq)

    # Pairwise squared Euclidean distance
    dist_sq = (
        norm_sq.unsqueeze(1)
        + norm_sq.unsqueeze(0)
        - 2 * torch.matmul(Y, Y.T)
    )

    # Gaussian kernel
    K = torch.exp(-gamma * dist_sq)

    return K
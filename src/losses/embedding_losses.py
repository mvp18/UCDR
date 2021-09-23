import torch
import torch.nn as nn
import torch.nn.functional as F


# Glove/W2V based CE-loss

def Cosine_CCE(semantic_emb):

    semantic_emb_norm = F.normalize(semantic_emb, p=2, dim=1)

    def cce_loss(enc_out, y_true):

        enc_out_norm = F.normalize(enc_out, p=2, dim=1)
        cosine_dist_logits = torch.mm(enc_out_norm, semantic_emb_norm.t())

        # y_true = y_true.type(torch.cuda.LongTensor) # y_true is a class index tensor of type long

        cross_entropy_loss = nn.CrossEntropyLoss()(cosine_dist_logits, y_true) # Softmax inside loss function

        return cross_entropy_loss

    return cce_loss


def Mixup_Cosine_CCE(semantic_emb):

    semantic_emb_norm = F.normalize(semantic_emb, p=2, dim=1)

    def mix_cce_loss(enc_out, y_a, y_b, ratios):

        enc_out_norm = F.normalize(enc_out, p=2, dim=1)
        cosine_dist_logits = torch.mm(enc_out_norm, semantic_emb_norm.t())

        cce_a = nn.CrossEntropyLoss(reduction='none')(cosine_dist_logits, y_a)
        cce_b = nn.CrossEntropyLoss(reduction='none')(cosine_dist_logits, y_b)

        mixup_cce = ratios*cce_a + (1 - ratios)*cce_b

        return mixup_cce.mean()

    return mix_cce_loss


def Euclidean_MSE(semantic_emb):

    def mse_loss(enc_out, cls_sim, cls_wts):

        eucdist_logits = torch.cdist(enc_out, semantic_emb, p=2.0)
        emb_mse = nn.MSELoss(reduction='none')(eucdist_logits, cls_sim)
        emb_mse *= cls_wts

        return emb_mse.mean()

    return mse_loss


def Mixup_Euclidean_MSE(semantic_emb, alpha):

    def mixup_mse_loss(enc_out, y_a, y_b, ratios):

        mixed_semantic_vec = ratios.unsqueeze(-1)*semantic_emb[y_a, :] + (1 - ratios.unsqueeze(-1))*semantic_emb[y_b, :]        

        gt_cls_sim = torch.cdist(mixed_semantic_vec, semantic_emb, p=2.0)
        eucdist_logits = torch.cdist(enc_out, semantic_emb, p=2.0)

        cls_euc_scaled = gt_cls_sim/(torch.max(gt_cls_sim, dim=1).values.unsqueeze(-1))
        cls_wts = torch.exp(-alpha*cls_euc_scaled)

        emb_mse = nn.MSELoss(reduction='none')(eucdist_logits, gt_cls_sim)
        emb_mse *= cls_wts

        return emb_mse.mean()

    return mixup_mse_loss
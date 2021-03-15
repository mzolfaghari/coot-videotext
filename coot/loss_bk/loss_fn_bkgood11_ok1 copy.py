"""
Loss functions.
"""
import time 
from typing import Callable, Dict
import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F

from nntrainer import typext
from nntrainer.typext import INF
from coot.clustering import train_kmeans, compute_cluster_assignment, FaissKMeans, MemoryReserver


class LossesConst(typext.ConstantHolder):
    CONTRASTIVE = "contrastive"
    CROSSENTROPY = "crossentropy"


def cosine_sim(visual_emb: th.Tensor, text_emb: th.Tensor) -> th.Tensor:
    """
    Calculate cosine similarity.

    Args:
        visual_emb: Visual embedding with shape (num_datapoints, dim_embedding)
        text_emb: Text embedding with shape (num_datapoints, dim_embedding)

    Returns:
        Cosine similariies with shape (num_datapoints, num_datapoints)
    """
    return visual_emb.mm(text_emb.t())


class ContrastiveLossConfig(typext.ConfigClass):
    """
    Contrastive loss Configuration Class

    Args:
        config: Configuration dictionary to be loaded, saving part.
    """

    def __init__(self, config: Dict) -> None:
        self.margin: float = config.pop("margin")
        self.weight_high: float = config.pop("weight_high")
        self.weight_high_internal: float = config.pop("weight_high_internal")
        self.weight_low: float = config.pop("weight_low")
        self.weight_low_internal: float = config.pop("weight_low_internal")
        self.weight_context: float = config.pop("weight_context")
        self.weight_context_internal: float = config.pop("weight_context_internal")

class ContrastiveLoss(nn.Module):
    """
    Regular Contrastive Loss between 2 groups of embeddings
    """

    def __init__(self, margin: float, max_violation: bool = False, norm: bool = True, use_cuda: bool = True):
        super().__init__()
        self.margin = margin
        self.sim = cosine_sim
        self.norm = norm
        self.max_violation = max_violation
        self.use_cuda = use_cuda

    def forward(self, im, s, x=0, y=0):
        """
        Inputs shape (batch, embed_dim)

        Args:
            im: Visual embeddings (batch, embed_dim)
            s: Text embeddings (batch, embed_dim)

        Returns:
        """
        # compute image-sentence score matrix - how close is im(y) to s(x)
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals, where there is just the margin left
        mask: th.Tensor = th.eye(scores.shape[0]).bool()
        if self.use_cuda:
            mask = mask.cuda(non_blocking=True)
        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        if self.norm:
            return (cost_s.sum() + cost_im.sum()).div(im.shape[0] * s.shape[0])
        return cost_s.sum() + cost_im.sum()

class NSCLoss(nn.Module):

    def __init__(self, temperature, use_cosine_similarity):
        super(NSCLoss, self).__init__()
        self.temperature = temperature
        self.softmax = th.nn.Softmax(dim=-1)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = th.nn.CrossEntropyLoss(reduction="sum")
        self.pdist = pdist = nn.PairwiseDistance(p=2)


    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = th.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = th.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(th.bool)
        return mask.cuda(non_blocking=True)

    @staticmethod
    def _dot_simililarity(x, y):
        v = th.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def _normalize_max(self, x, axis=-1):
        # from https://github.com/LvWilliam/EWTH_Loss/blob/main/BoT/triplet_loss.py
        """Normalizing to unit length along the specified dimension.
        Args:
        x: pytorch Variable
        Returns:
        x: pytorch Variable, same shape as input
        """
        dis = th.sum(x.pow(2), dim=1).sqrt()
        m, _ = th.max(dis, 0)
        x = x / m
        return x

    def _normalize(self, x, axis=-1):
        """Normalizing to unit length along the specified dimension.
        Args:
        x: pytorch Variable
        Returns:
        x: pytorch Variable, same shape as input
        """
        x = 1. * x / (th.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
        return x
    # def forward(self, zjs, zis, x=0, y=0):
    #     # print("===> ", zjs.shape, zis.shape, x.shape, y.shape)
    #     logits_1 = self.forward_oneside(zjs, zis)
    #     logits_2 = self.forward_oneside(zis, zjs)
    #     labels = th.zeros(2 * self.batch_size).cuda(non_blocking=True).long()
    #     # print(labels)
    #     loss = (self.criterion(logits_1, labels) + self.criterion(logits_2, labels) ) /2
    #     return loss / (2 * zjs.shape[0])

    def forward(self, zjs, zis):
        zjs = self._normalize_max(zjs, axis=-1)
        zis = self._normalize_max(zis, axis=-1)

        representations = th.cat([zjs, zis], dim=0)
        self.batch_size = zis.shape[0]

        similarity_matrix = self.similarity_function(representations, representations)

        # print(similarity_matrix.shape)
        # filter out the scores from the positive samples
        l_pos = th.diag(similarity_matrix, self.batch_size)
        r_pos = th.diag(similarity_matrix, -self.batch_size)

        self.mask_samples_from_same_repr = self._get_correlated_mask().type(th.bool)
        positives = th.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = th.cat((positives, negatives), dim=1)
        # print(logits.shape, positives.shape, negatives.shape, zjs.shape, "==>")
        logits /= self.temperature
        labels = th.zeros(2 * self.batch_size).cuda(non_blocking=True).long()
        loss = self.criterion(logits, labels)
        # import pdb; pdb.set_trace()
        # pdist = self.pdist(zjs1, zis1)
        # cost_ml = pdist.mean()
        return loss / (2 * self.batch_size) #+  cost_ml

class NTXentLoss(nn.Module):

    def __init__(self, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.softmax = th.nn.Softmax(dim=-1)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = th.nn.CrossEntropyLoss(reduction="sum")
        self.pdist = pdist = nn.PairwiseDistance(p=2)


    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = th.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = th.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(th.bool)
        return mask.cuda(non_blocking=True)

    @staticmethod
    def _dot_simililarity(x, y):
        v = th.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def _normalize_max(self, x, axis=-1):
        # from https://github.com/LvWilliam/EWTH_Loss/blob/main/BoT/triplet_loss.py
        """Normalizing to unit length along the specified dimension.
        Args:
        x: pytorch Variable
        Returns:
        x: pytorch Variable, same shape as input
        """
        dis = th.sum(x.pow(2), dim=1).sqrt()
        m, _ = th.max(dis, 0)
        x = x / m
        return x

    def _normalize(self, x, axis=-1):
        """Normalizing to unit length along the specified dimension.
        Args:
        x: pytorch Variable
        Returns:
        x: pytorch Variable, same shape as input
        """
        x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
        return x
    def forward(self, zjs, zis, x=0, y=0):
        logits_1 = self.forward_oneside(zjs, zis)
        # logits_2 = self.forward_oneside(zis, zjs)
        labels = th.zeros(2 * self.batch_size).cuda(non_blocking=True).long()
        loss = self.criterion(logits_1, labels) #+ self.criterion(logits_2, labels) ) /2
        return loss 

    def forward_oneside(self, zjs, zis):
        # zjs = self._normalize_max(zjs1, axis=-1)
        # zis = self._normalize_max(zis1, axis=-1)
        representations = th.cat([zjs, zis], dim=0)
        self.batch_size = zis.shape[0]

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = th.diag(similarity_matrix, self.batch_size)
        r_pos = th.diag(similarity_matrix, -self.batch_size)

        self.mask_samples_from_same_repr = self._get_correlated_mask().type(th.bool)
        positives = th.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = th.cat((positives, negatives), dim=1)
        logits /= self.temperature


        # pdist = self.pdist(zjs1, zis1)
        # cost_ml = pdist.mean()
        return logits #loss / (2 * self.batch_size) #+  cost_ml

class MILNCELoss(th.nn.Module):
    def __init__(self):
        super(MILNCELoss, self).__init__()

    def forward(self, video_embd, text_embd, x=0, y=0):
        x = th.matmul(video_embd, text_embd.t())
        x = x.view(video_embd.shape[0], video_embd.shape[0], -1)
        nominator = x * th.eye(x.shape[0])[:,:,None].cuda()
        nominator = nominator.sum(dim=1)
        nominator = th.logsumexp(nominator, dim=1)
        denominator = th.cat((x, x.permute(1,0,2)), dim=1).view(x.shape[0], -1)
        denominator = th.logsumexp(denominator, dim=1)
        return th.mean(denominator - nominator)

class DCL(th.nn.Module):
    def __init__(self):
        super(DCL, self).__init__()

    def get_negative_mask(self, batch_size):
        negative_mask = th.ones((batch_size, 2 * batch_size), dtype=bool)
        for i in range(batch_size):
            negative_mask[i, i] = 0
            negative_mask[i, i + batch_size] = 0

        negative_mask = th.cat((negative_mask, negative_mask), 0)
        return negative_mask

    def forward(self, video_embd, text_embd, x=0, y=0):
        batch_size = video_embd.shape[0]
        temperature = 0.1
        tau_plus = 0.01
        debiased = 1
       # neg score
        out = th.cat([video_embd, text_embd], dim=0)
        neg = th.exp(th.mm(out, out.t().contiguous()) / temperature)
        mask = self.get_negative_mask(batch_size).cuda()
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        pos = th.exp(th.sum(video_embd * text_embd, dim=-1) / temperature)
        pos = th.cat([pos, pos], dim=0)

        # estimator g()
        if debiased:
            N = batch_size * 2 - 2
            Ng = (-tau_plus * N * pos + neg.sum(dim = -1)) / (1 - tau_plus)
            # constrain (optional)
            Ng = th.clamp(Ng, min = N * np.e**(-1 / temperature))
        else:
            Ng = neg.sum(dim=-1)

        # contrastive loss
        loss = (- th.log(pos / (pos + Ng) )).mean()
        return loss

class CrossCLR(nn.Module):
    """
    CLIP Loss between 2 groups of embeddings
    """

    def __init__(self, temperature=0.05, logger = None):
        super().__init__()
        self.logit_scale = nn.Parameter(th.ones([]))
        # self.criterion = th.nn.CrossEntropyLoss(reduction="sum")
        self.criterion = th.nn.CrossEntropyLoss(reduction='none') #th.nn.CrossEntropyLoss()
        self.temperature = 0.03 #temperature #0.04
        self.logger = logger
        self.faiss_memory = MemoryReserver()
        self.nsc = NSCLoss(0.09, 1)
        self.K = 3000
        dim = 384 #network embedding size
        dim_org_vid = 512
        dim_org_txt = 1536

        # create the queue
        self.register_buffer("queue_vid", th.randn(dim, self.K ))
        self.register_buffer("queue_txt", th.randn(dim, self.K ))
        self.register_buffer("queue_vid_org", th.randn(dim_org_vid, self.K ))
        self.register_buffer("queue_txt_org", th.randn(dim_org_txt, self.K ))

        self.queue_vid = nn.functional.normalize(self.queue_vid, dim=0)
        self.queue_txt = nn.functional.normalize(self.queue_txt, dim=0)
        self.queue_vid_org = nn.functional.normalize(self.queue_vid_org, dim=0)
        self.queue_txt_org = nn.functional.normalize(self.queue_txt_org, dim=0)

        self.register_buffer("queue_ptr", th.zeros(1, dtype=th.long))


    def _get_positive_mask(self, batch_size):
        diag = np.eye(batch_size)
        mask = th.from_numpy((diag))
        mask = (1 - mask)
        return mask.cuda(non_blocking=True)

    def _get_positive_mask_bank(self, k, batch_size, ptr):
        diag = np.eye(batch_size)
        mask = th.from_numpy((diag))
        # mask = (1 - mask)

        diag_bank = np.ones((batch_size, k))
        mask_bank = th.from_numpy((diag_bank))

        if (ptr+batch_size) > k:
            qptr_end = k
            inp_feat_k = batch_size - (ptr+batch_size - k)
            mask_bank[:, ptr:] -= mask[:,:inp_feat_k]
        else:
            mask_bank[:, ptr:ptr+batch_size] -= mask

        return mask_bank.cuda(non_blocking=True)

    @th.no_grad()
    def _dequeue_and_enqueue(self, keys_vid, keys_txt, keys_vid_org, keys_txt_org):


        batch_size = keys_vid.shape[0]

        ptr = int(self.queue_ptr)
        # print(self.K, batch_size)
        # assert self.K % batch_size == 0  # for simplicity
        
        # replace the keys at ptr (dequeue and enqueue)
        ptr_end = ptr+batch_size
        ptr_end_k = batch_size
        if (ptr+batch_size) > self.K:
            ptr_end = self.K
            ptr_end_k = self.K - (ptr + batch_size)
        # print(ptr_end, ptr_end_k)
        # import pdb; pdb.set_trace()
        self.queue_vid[:, ptr:ptr_end] = keys_vid.T[:, :ptr_end_k]
        self.queue_txt[:, ptr:ptr_end] = keys_txt.T[:, :ptr_end_k]
        self.queue_vid_org[:, ptr:ptr_end] = keys_vid_org.T[:, :ptr_end_k]
        self.queue_txt_org[:, ptr:ptr_end] = keys_txt_org.T[:, :ptr_end_k]
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, image_features, text_features, input_img, input_txt):
        """
        Inputs shape (batch, embed_dim)

        Args:
            im: Visual embeddings (batch, embed_dim)
            s: Text embeddings (batch, embed_dim)

        Returns:
        """

        # ================= Params ==========================
        weighted_loss = 1 #Active or deactivate loss weightening.
        intra_modality = True #Use intra modality scores in contrastive objective
        no_prune = 1 #0: no pruning, 1: threshold, 2: percentage
        thrshold = 0.6#0.8

        # =====
        ## Tomorrow test with removing lines 412 and 414! Normalization might remove some info!
        # also have look on normaliztion line 709, 710
        #======
        # thrsh_img = 0.8 #0.012 #threshold for image similarity - find influential samples
        # thrsh_txt = 170 #threshold for text similarity - find influential samples

        cls_method = 6 # Method 5 or 6! Basically same but pruning top or bottom percent of samples

        use_emb = 0 # Use network embeddings or original embeddings to weight loss
        no_softmax = True # Apply softmax on logits before feeding them to crossentropy loss
        w_temp = 0.0035 # Temperature for scaling weights. self.logit_scale # Naming was wrong 1e1 means 1e2!!
        prune_percent = 0.25 # percentage of pruning negatives ! This will be used only for no_prune=2
        # In some of experiments I wrongly used prune percent instead of weight negative
        w_negative = 0.8 #0.8 weight of negative scores.


        input_img1 = th.mean(input_img, dim=1)
        input_txt1 = th.mean(input_txt, dim=1)
        # This a new normalization for input features! We didn't have for experiments < 9 Feb 2021
        input_img2 = input_img1 #/ input_img1.max(dim=-1, keepdim=True)[0]
        # import pdb; pdb.set_trace()
        input_txt2 = input_txt1 #/ input_txt1.max(dim=-1, keepdim=True)[0]
        # input_img2 = input_txt2
        # input_img = input_txt
        # import pdb; pdb.set_trace()

        # ===================================================
        # normalized features
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = nn.functional.normalize(image_features, dim=1)
        text_features = nn.functional.normalize(text_features, dim=1)

        self._dequeue_and_enqueue(image_features, text_features, input_img2, input_txt2)

        logits_per_image = image_features @ text_features.t()
        logits_per_text = text_features @ image_features.t()
        # logits_clstr_img = image_features @ image_features.t()
        # logits_clstr_txt = text_features @ text_features.t()

        logits_clstr_img = image_features @ self.queue_vid.cuda()
        logits_clstr_txt = text_features @ self.queue_txt.cuda()

        logits_per_image /= self.temperature #* self.logit_scale.exp()
        logits_per_text /= self.temperature #* self.logit_scale.exp()
        logits_clstr_img /= self.temperature #* self.logit_scale.exp()
        logits_clstr_txt /= self.temperature #* self.logit_scale.exp()

        qptr_end = self.queue_ptr[0]+image_features.shape[0]
        if (self.queue_ptr[0]+image_features.shape[0]) > self.K:
            qptr_end = self.K
            inp_feat_k = image_features.shape[0] - (self.queue_ptr[0]+image_features.shape[0] - self.K)
            positive_mask =  self._get_positive_mask_bank(self.K, image_features.shape[0], self.queue_ptr[0])
            # import pdb; pdb.set_trace()
            # sim_scores_img = (input_img2[:inp_feat_k,:] @ self.queue_vid_org.cuda()) * positive_mask
            # sim_scores_txt= (input_txt2[:inp_feat_k,:] @ self.queue_txt_org.cuda()) * positive_mask
            # logits_per_image = logits_per_image[:inp_feat_k,:]
            # logits_per_text = logits_per_text[:inp_feat_k,:]
            # logits_clstr_img = logits_clstr_img[:inp_feat_k,:]
            # logits_clstr_txt = logits_clstr_txt[:inp_feat_k,:]
        else:
            positive_mask =  self._get_positive_mask_bank(self.K, image_features.shape[0], self.queue_ptr[0]) #self._get_positive_mask(self.K)[self.queue_ptr[0]:qptr_end,:]
            
        sim_scores_img = (input_img2 @ self.queue_vid_org.cuda()) * positive_mask
        sim_scores_txt= (input_txt2 @ self.queue_txt_org.cuda()) * positive_mask
            

        

        if weighted_loss == 0:
            # ---- CLIP ---
            # logits_per_image = logits_per_image / 0.05 #self.logit_scale.exp()
            # logits_per_text = logits_per_text / 0.05 #self.logit_scale.exp()
            # logits_clstr_img = self.logit_scale.exp()
            # logits_clstr_txt = self.logit_scale.exp()
            positive_mask =  self._get_positive_mask(image_features.shape[0]).type(th.bool)
            logits_clstr_img = logits_clstr_img * positive_mask
            logits_clstr_txt = logits_clstr_txt * positive_mask
            labels = th.arange(logits_per_image.shape[0]).cuda()
            

            if intra_modality:
                if no_softmax:  
                    img_logits_prune = th.cat([logits_per_image, w_negative * logits_clstr_img], dim=1)
                    txt_logits_prune = th.cat([logits_per_text, w_negative * logits_clstr_txt], dim=1)
                else: 
                    img_logits_prune = th.cat([F.softmax(logits_per_image, dim=1), w_negative * F.softmax(logits_clstr_img, dim=1)], dim=1)
                    txt_logits_prune = th.cat([F.softmax(logits_per_text, dim=1), w_negative * F.softmax(logits_clstr_txt, dim=1)], dim=1)
            else:
                if no_softmax:  
                    img_logits_prune = logits_per_image
                    txt_logits_prune = logits_per_text
                else: 
                    img_logits_prune = F.softmax(logits_per_image, dim=1)
                    txt_logits_prune = F.softmax(logits_per_text, dim=1)

            loss_i = self.criterion(img_logits_prune, labels)
            loss_t = self.criterion(txt_logits_prune, labels)
            loss_i = loss_i.mean()
            loss_t = loss_t.mean()

        elif weighted_loss == 1:
            # logits_clstr_img = image_features @ image_features.t()
            # logits_clstr_txt = text_features @ text_features.t()

            # logits_per_image /= self.temperature #* self.logit_scale.exp()
            # logits_per_text /= self.temperature #* self.logit_scale.exp()
            # logits_clstr_img /= self.temperature #* self.logit_scale.exp()
            # logits_clstr_txt /= self.temperature #* self.logit_scale.exp()
            # logits = torch.matmul(img_feats, text_feats.T) * torch.exp(self.temperature_factor)
            if use_emb:
                input_img = image_features
                input_txt = text_features

            # positive_mask =  self._get_positive_mask(image_features.shape[0])
            # qptr_end = self.queue_ptr[0]+image_features.shape[0]
            # if (self.queue_ptr[0]+image_features.shape[0]) > self.K:
            #     qptr_end = self.K
            #     inp_feat_k = image_features.shape[0] - (self.queue_ptr[0]+image_features.shape[0] - self.K)
            #     positive_mask =  self._get_positive_mask(self.K)[self.queue_ptr[0]:qptr_end,:]
            #     sim_scores_img = (input_img2[:inp_feat_k,:] @ self.queue_vid_org.cuda()) * positive_mask
            #     sim_scores_txt= (input_txt2[:inp_feat_k,:] @ self.queue_txt_org.cuda()) * positive_mask
            # else:
            #     positive_mask =  self._get_positive_mask(self.K)[self.queue_ptr[0]:qptr_end,:]
            #     sim_scores_img = (input_img2 @ self.queue_vid_org.cuda()) * positive_mask
            #     sim_scores_txt= (input_txt2 @ self.queue_txt_org.cuda()) * positive_mask

            # sim_scores_img = (input_img2 @ input_img2.t()) * positive_mask
            # sim_scores_txt= (input_txt2 @ input_txt2.t()) * positive_mask
            
            # try:
            #     sim_scores_img = (input_img2 @ self.queue_vid_org.cuda()) * positive_mask
            #     sim_scores_txt= (input_txt2 @ self.queue_txt_org.cuda()) * positive_mask
            # except:
            #     import pdb; pdb.set_trace()

            ##----- Dist ----
            # sim_scores_img = (logits_per_image) * positive_mask
            # sim_scores_txt= (logits_per_text) * positive_mask

            #===============
            # positive_mask =  self._get_positive_mask(image_features.shape[0]).type(th.bool)
            # sim_scores_img = sim_scores_img * positive_mask
            # sim_scores_txt = sim_scores_txt * positive_mask
            avg_sim_img = th.mean(sim_scores_img,dim=1)
            avg_sim_txt = th.mean(sim_scores_txt,dim=1)
            

            sorted_img, indices_img = th.sort(avg_sim_img)
            sorted_txt, indices_txt = th.sort(avg_sim_txt)
            sorted_img = sorted_img / sorted_img.max(dim=-1, keepdim=True)[0]
            sorted_txt = sorted_txt / sorted_txt.max(dim=-1, keepdim=True)[0]


            # ======================================================
            # Find index of influential samples and remove them from negative set
            indices_img_thrsh = indices_img[sorted_img<thrshold]
            indices_txt_thrsh = indices_txt[sorted_txt<thrshold]
            # indices_img_thrsh = indices_txt_thrsh

            # # import pdb; pdb.set_trace()
            # print("Image:", sorted_img[1],sorted_img[-1])  #0.012
            # print("Text:", sorted_txt[1],sorted_txt[-1])   #90
            # print("==="*20)    
            labels = th.arange(image_features.shape[0]).cuda()
            # print(logits_per_image.shape)

            # print(labels)
            # img_logits_prune = logits_per_image[:, indices_txt[n_prune:]][indices_txt[n_prune:],:]
            # txt_logits_prune = logits_per_text[:, indices_img[n_prune:]][indices_img[n_prune:],:]
            if cls_method == 1:
                img_logits_prune = logits_per_image[:, indices_txt[:-n_prune]][indices_txt[:-n_prune],:]
                txt_logits_prune = logits_per_text[:, indices_img[:-n_prune]][indices_img[:-n_prune],:]
            elif cls_method == 2:
                img_logits_prune = logits_per_image[:, indices_txt[n_prune:]][indices_txt[n_prune:],:]
                txt_logits_prune = logits_per_text[:, indices_img[n_prune:]][indices_img[n_prune:],:]
            elif cls_method == 3:
                img_logits_prune = (F.softmax(logits_per_image, dim=1) + 0.9 * F.softmax(logits_clstr_img, dim=1))/2
                txt_logits_prune = (F.softmax(logits_per_text, dim=1)  + 0.9 * F.softmax(logits_clstr_txt, dim=1))/2

            elif cls_method == 4:
                # positive_mask =  self._get_positive_mask(image_features.shape[0])
                # --------- loss_CrossCLR_RemClstrPos_r4 -----
                positive_mask =  self._get_positive_mask(image_features.shape[0]).type(th.bool)
                negatives_img = logits_clstr_img[positive_mask].view(image_features.shape[0], -1)
                negatives_txt = logits_clstr_txt[positive_mask].view(image_features.shape[0], -1)
                # negatives_img = logits_clstr_img * positive_mask
                # negatives_txt = logits_clstr_txt * positive_mask
                # import pdb; pdb.set_trace()
                if no_softmax:
                    img_logits_prune = th.cat([logits_per_image, negatives_img], dim=1)
                    txt_logits_prune = th.cat([logits_per_text, negatives_txt], dim=1)  
                else:
                    # img_logits_prune = th.cat([F.softmax(logits_per_image, dim=1), F.softmax(logits_clstr_img, dim=1)], dim=1)
                    # txt_logits_prune = th.cat([F.softmax(logits_per_text, dim=1), F.softmax(logits_clstr_txt, dim=1)], dim=1)  
                    img_logits_prune = th.cat([F.softmax(logits_per_image, dim=1), F.softmax(negatives_img, dim=1)], dim=1)
                    txt_logits_prune = th.cat([F.softmax(logits_per_text, dim=1), F.softmax(negatives_txt, dim=1)], dim=1) 

            elif cls_method == 5:

                # --------- loss_CrossCLR_RemClstrPos_Prune -----
                n_prune = int((logits_clstr_img.shape[0]  - prune_percent *  logits_clstr_img.shape[0]) // 1)
                positive_mask =  self._get_positive_mask(image_features.shape[0]).type(th.bool)
                logits_clstr_img = logits_clstr_img * positive_mask
                logits_clstr_txt = logits_clstr_txt * positive_mask

                if no_prune:
                    negatives_img = logits_clstr_img
                    negatives_txt = logits_clstr_txt
                else:
                    negatives_img = logits_clstr_img[:, indices_img[n_prune:]]
                    negatives_txt = logits_clstr_txt[:, indices_txt[n_prune:]]
                
                # positive_mask_img = positive_mask[:, indices_img[n_prune:]]
                # positive_mask_txt = positive_mask[:, indices_txt[n_prune:]]
                # negatives_img = logits_clstr_img[positive_mask_img].view(image_features.shape[0], -1)
                # negatives_txt = logits_clstr_txt[positive_mask_txt].view(image_features.shape[0], -1)

                # img_logits_prune = th.cat([F.softmax(logits_per_image, dim=1), F.softmax(logits_clstr_img, dim=1)], dim=1)
                # txt_logits_prune = th.cat([F.softmax(logits_per_text, dim=1), F.softmax(logits_clstr_txt, dim=1)], dim=1)
                if intra_modality:
                    if no_softmax:  
                        img_logits_prune = th.cat([logits_per_image, w_negative * negatives_img], dim=1)
                        txt_logits_prune = th.cat([logits_per_text, w_negative * negatives_txt], dim=1) 
                    else:
                        img_logits_prune = th.cat([F.softmax(logits_per_image, dim=1), w_negative * F.softmax(negatives_img, dim=1)], dim=1)
                        txt_logits_prune = th.cat([F.softmax(logits_per_text, dim=1), w_negative * F.softmax(negatives_txt, dim=1)], dim=1)
                else:
                    if no_softmax:  
                        img_logits_prune = logits_per_image
                        txt_logits_prune = logits_per_text
                    else:
                        img_logits_prune = F.softmax(logits_per_image, dim=1)
                        txt_logits_prune = F.softmax(logits_per_text, dim=1)

                # img_logits_prune = th.cat([logits_per_image, 0.8 * negatives_img], dim=1)
                # txt_logits_prune = th.cat([logits_per_text, 0.8 * negatives_txt], dim=1) 


            elif cls_method == 6:
                # --------- loss_CrossCLR_RemClstrPos_Prune -----
                # n_prune = int((logits_clstr_img.shape[0]  - prune_percent *  logits_clstr_img.shape[0]) // 1)
                # n_prune = (logits_clstr_img.shape[0] - 200
                # positive_mask =  self._get_positive_mask(logits_clstr_img.shape[0]).type(th.bool)
                # positive_mask =  self._get_positive_mask(self.K)[self.queue_ptr[0]:self.queue_ptr[0]+logits_clstr_img.shape[0],:].type(th.bool)
                logits_clstr_img = logits_clstr_img * positive_mask
                logits_clstr_txt = logits_clstr_txt * positive_mask

                if no_prune == 0:
                    negatives_img = logits_clstr_img
                    negatives_txt = logits_clstr_txt

                elif no_prune == 1:
                    # import pdb; pdb.set_trace()
                    # negatives_img = logits_clstr_img[:, indices_img_thrsh]
                    # negatives_txt = logits_clstr_txt[:, indices_img_thrsh]
                    negatives_img = logits_clstr_img#[:, indices_img_thrsh]
                    negatives_txt = logits_clstr_txt#[:, indices_txt_thrsh]

                else:
                    negatives_img = logits_clstr_img[:, indices_img[:n_prune]]
                    negatives_txt = logits_clstr_txt[:, indices_txt[:n_prune]]


                # img_logits_prune = th.cat([F.softmax(logits_per_image, dim=1), F.softmax(logits_clstr_img, dim=1)], dim=1)
                # txt_logits_prune = th.cat([F.softmax(logits_per_text, dim=1), F.softmax(logits_clstr_txt, dim=1)], dim=1) 
                if intra_modality:
                    if no_softmax:  
                        # img_logits_prune = th.cat([logits_per_image, w_negative * negatives_img, w_negative * negatives_txt], dim=1)
                        # txt_logits_prune = th.cat([logits_per_text, w_negative * negatives_txt, w_negative * negatives_img], dim=1)
                        img_logits_prune = th.cat([logits_per_image, w_negative * negatives_img], dim=1)
                        txt_logits_prune = th.cat([logits_per_text, w_negative * negatives_txt], dim=1)
                    else: 
                        # img_logits_prune = th.cat([F.softmax(logits_per_image, dim=1), w_negative * F.softmax(negatives_img, dim=1), w_negative * F.softmax(negatives_txt, dim=1)], dim=1)
                        # txt_logits_prune = th.cat([F.softmax(logits_per_text, dim=1), w_negative * F.softmax(negatives_txt, dim=1), w_negative * F.softmax(negatives_img, dim=1)], dim=1)
                        img_logits_prune = th.cat([F.softmax(logits_per_image, dim=1), w_negative * F.softmax(negatives_img, dim=1)], dim=1)
                        txt_logits_prune = th.cat([F.softmax(logits_per_text, dim=1), w_negative * F.softmax(negatives_txt, dim=1)], dim=1)
                
                else:
                    if no_softmax:  
                        img_logits_prune = logits_per_image
                        txt_logits_prune = logits_per_text
                    else: 
                        img_logits_prune = F.softmax(logits_per_image, dim=1)
                        txt_logits_prune = F.softmax(logits_per_text, dim=1)
                
                labels_prune = th.arange(logits_per_image.shape[0]).cuda()
                
                loss_i = self.criterion(img_logits_prune, labels_prune)
                loss_t = self.criterion(txt_logits_prune, labels_prune)

            else:
                # --------- loss_CrossCLR_RemClstrPos_Prune -----
                n_prune = int((logits_clstr_img.shape[0]  - prune_percent *  logits_clstr_img.shape[0]) // 1)
                # n_prune = (logits_clstr_img.shape[0] - 200
                positive_mask =  self._get_positive_mask(logits_clstr_img.shape[0]).type(th.bool)
                logits_clstr_img = logits_clstr_img * positive_mask
                logits_clstr_txt = logits_clstr_txt * positive_mask

                if no_prune == 0:
                    negatives_img = logits_clstr_img
                    negatives_txt = logits_clstr_txt

                elif no_prune == 1:
                    negatives_img = logits_clstr_img[:, indices_img_thrsh]
                    negatives_txt = logits_clstr_txt[:, indices_img_thrsh]                    
                else:
                    negatives_img = logits_clstr_img[:, indices_img[:n_prune]]
                    negatives_txt = logits_clstr_txt[:, indices_txt[:n_prune]]

                # img_logits_prune = th.cat([F.softmax(logits_per_image, dim=1), F.softmax(logits_clstr_img, dim=1)], dim=1)
                # txt_logits_prune = th.cat([F.softmax(logits_per_text, dim=1), F.softmax(logits_clstr_txt, dim=1)], dim=1) 
                if intra_modality:
                    if no_softmax:  
                        # img_logits_prune = th.cat([logits_per_image, w_negative * negatives_img, w_negative * negatives_txt], dim=1)
                        # txt_logits_prune = th.cat([logits_per_text, w_negative * negatives_txt, w_negative * negatives_img], dim=1)
                        img_logits_prune1 = logits_per_image
                        img_logits_prune2 = th.cat([th.diag(logits_per_image).view(logits_per_image.shape[0], 1), w_negative * negatives_img], dim=1)
                        txt_logits_prune1 = logits_per_text
                        txt_logits_prune2 = th.cat([th.diag(logits_per_text).view(logits_per_image.shape[0], 1), w_negative * negatives_txt], dim=1)
                    else: 
                        # img_logits_prune = th.cat([F.softmax(logits_per_image, dim=1), w_negative * F.softmax(negatives_img, dim=1), w_negative * F.softmax(negatives_txt, dim=1)], dim=1)
                        # txt_logits_prune = th.cat([F.softmax(logits_per_text, dim=1), w_negative * F.softmax(negatives_txt, dim=1), w_negative * F.softmax(negatives_img, dim=1)], dim=1)
                        img_logits_prune = th.cat([F.softmax(logits_per_image, dim=1), w_negative * F.softmax(negatives_img, dim=1)], dim=1)
                        txt_logits_prune = th.cat([F.softmax(logits_per_text, dim=1), w_negative * F.softmax(negatives_txt, dim=1)], dim=1)
                
                    labels_prune1 = th.arange(logits_per_image.shape[0]).cuda()
                    labels_prune2 = th.zeros(logits_per_image.shape[0]).cuda(non_blocking=True).long()
                    
                    loss_i = (self.criterion(img_logits_prune1, labels_prune1) + self.criterion(img_logits_prune2, labels_prune2))
                    loss_t = (self.criterion(txt_logits_prune1, labels_prune1) +  self.criterion(txt_logits_prune2, labels_prune2))
                else:
                    if no_softmax:  
                        img_logits_prune = logits_per_image
                        txt_logits_prune = logits_per_text
                    else: 
                        img_logits_prune = F.softmax(logits_per_image, dim=1)
                        txt_logits_prune = F.softmax(logits_per_text, dim=1)    

                    labels_prune = th.arange(logits_per_image.shape[0]).cuda()
                    
                    loss_i = self.criterion(img_logits_prune, labels_prune)
                    loss_t = self.criterion(txt_logits_prune, labels_prune)   

                # img_logits_prune = th.cat([logits_per_image, 0.8 * negatives_img], dim=1)
                # txt_logits_prune = th.cat([logits_per_text, 0.8 * negatives_txt], dim=1)   
                # print(logits_per_image.shape, negatives_img.shape, "=====>")
              
                # img_logits_prune = (F.softmax(logits_per_image, dim=1) + 0.8 * F.softmax(logits_clstr_img, dim=1)) / 2
                # txt_logits_prune = (F.softmax(logits_per_text, dim=1) + 0.8 * F.softmax(logits_clstr_txt, dim=1)) / 2

                # --- In new_loss3 I wrongly name MergeNoSoftmax! but it's not merging! its cat
                # img_logits_prune = th.cat([logits_per_image, 0.8 * logits_clstr_img], dim=1)
                # txt_logits_prune = th.cat([logits_per_text, 0.8 * logits_clstr_txt], dim=1) 



            w_i =  ((avg_sim_img/sum(avg_sim_img)))
            w_t = ((avg_sim_txt/sum(avg_sim_txt)))
            loss_i = loss_i * th.exp(w_i/ w_temp)
            loss_t = loss_t * th.exp(w_t/ w_temp)


            loss_i = sum(loss_i) / (sum(th.exp(w_i/ w_temp)))
            loss_t = sum(loss_t) / (sum(th.exp(w_t/ w_temp)))
            if weighted_loss == 0:
                raise("something is wrong!")
                


        loss = ((loss_i + loss_t)  )  / 2

        # loss = (loss_i.mean() + loss_t.mean()) / 2.
        return loss

class CLIP_Cluster(nn.Module):
    """
    CLIP Loss between 2 groups of embeddings
    """

    def __init__(self, temperature=0.05, logger = None):
        super().__init__()
        self.logit_scale = nn.Parameter(th.ones([]))
        # self.criterion = th.nn.CrossEntropyLoss(reduction="sum")
        self.criterion = th.nn.CrossEntropyLoss(reduction='none') #th.nn.CrossEntropyLoss()
        self.temperature = temperature #0.04
        self.logger = logger
        self.faiss_memory = MemoryReserver()
        self.nsc = NSCLoss(0.09, 1)

    def _get_positive_mask(self, batch_size):
        diag = np.eye(batch_size)
        mask = th.from_numpy((diag))
        mask = (1 - mask)
        return mask.cuda(non_blocking=True)

    def forward(self, image_features, text_features, input_img, input_txt):
        """
        Inputs shape (batch, embed_dim)

        Args:
            im: Visual embeddings (batch, embed_dim)
            s: Text embeddings (batch, embed_dim)

        Returns:
        """
        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # self.logger.debug('FAISS: clustering...')
        t0 = time.time()

        # input_features = th.mean(input_img, dim=1)
        # print(image_features.shape,input_features.shape )
        
        # centroids = train_kmeans(input_features, num_clusters=num_clusters, gpu_ids = 0, niter=100, nredo=1, verbose=0)
        # labels = compute_cluster_assignment(centroids, input_features.cpu().data.numpy())
        #=======================================
        prune_cluster = 0
        if prune_cluster == 1:
            prune_percent = 0.1
            input_features = th.mean(input_img, dim=1)
            num_clusters = int((input_features.shape[0]  - prune_percent *  input_features.shape[0]) // 1)
            self.faiss_memory.lock("faiss-gpu")
            print(image_features.shape, input_features.shape , num_clusters)
            faiss_ob = FaissKMeans(n_clusters=num_clusters, n_init=10, max_iter=100)
            centroids = faiss_ob.fit(input_features.cpu().data.numpy())
            labels_cls = faiss_ob.predict(input_features.cpu().data.numpy())
            print(labels_cls)
            import pdb; pdb.set_trace()
            # centroids = train_kmeans(input_features.cpu().data.numpy(), num_clusters=num_clusters, gpu_ids = 0, niter=100, nredo=1, verbose=0)
            # labels_cls = compute_cluster_assignment(centroids, input_features.cpu().data.numpy())
            self.faiss_memory.release()
        #========================================

        # sim_scores = input_features @ input_features.t()
        # import pdb 
        # pdb.set_trace()
        
        # labels = th.zeros((input_features.shape[0]), dtype=th.long)
        # for i, lblx in enumerate(labels_cls):
        #     # print(i, lblx)
        #     labels[i] = th.from_numpy(np.array(lblx))
        # labels = labels.cuda()
        # # print(labels)
        # t1 = time.time()
        # import pdb 
        # pdb.set_trace()

        # self.logger.debug("FAISS: Clustering total elapsed time: %.3f m" % ((t1 - t0) / 60.0))

        # cosine similarity as logits
        # logit_scale = self.logit_scale.exp()
        # logits_per_iamge = logit_scale * image_features @ text_features.t()
        # logits_per_text = logit_scale * text_features @ image_features.t()

        
        logits_per_image = image_features @ text_features.t()
        logits_per_text = text_features @ image_features.t()
        logits_clstr_img = image_features @ image_features.t()
        logits_clstr_txt = text_features @ text_features.t()

        logits_per_image /= self.temperature #* self.logit_scale.exp()
        logits_per_text /= self.temperature #* self.logit_scale.exp()
        logits_clstr_img /= self.temperature #* self.logit_scale.exp()
        logits_clstr_txt /= self.temperature #* self.logit_scale.exp()

        weighted_loss = 1

        if weighted_loss == 0:
            logits_per_image = F.softmax(logits_per_image, dim=1) 
            logits_per_text = F.softmax(logits_per_text, dim=1)
            labels = th.arange(logits_per_image.shape[0]).cuda()
            
            loss_i = self.criterion(logits_per_image, labels)
            loss_t = self.criterion(logits_per_text, labels)
            loss_i = loss_i.mean()
            loss_t = loss_t.mean()

        elif weighted_loss == 1:

            # logits = torch.matmul(img_feats, text_feats.T) * torch.exp(self.temperature_factor)
            input_img = th.mean(input_img, dim=1)
            input_txt = th.mean(input_txt, dim=1)
            positive_mask =  self._get_positive_mask(image_features.shape[0])
            sim_scores_img = (input_img @ input_img.t()) * positive_mask
            sim_scores_txt= (input_txt @ input_txt.t()) * positive_mask

            ##----- Dist ----
            # sim_scores_img = (logits_per_image) * positive_mask
            # sim_scores_txt= (logits_per_text) * positive_mask

            #===============
            avg_sim_img = th.mean(sim_scores_img,dim=1)
            avg_sim_txt = th.mean(sim_scores_txt,dim=1)
            sorted_img, indices_img = th.sort(avg_sim_img)
            sorted_txt, indices_txt = th.sort(avg_sim_txt)     
            labels = th.arange(image_features.shape[0]).cuda()
            # print(logits_per_image.shape)
            # import pdb
            # pdb.set_trace()
            #==============================
            # x = th.Tensor(46, 300)
            # print(x.size()) #torch.Size([46, 300])

            # first_half = x[0:20, :]
            # second_half = x[20:, :]
            # new_row = torch.Tensor(1, 300)
            # new_x = torch.cat(first_half, new_row, second_half)
            #==============================
            # print(labels)
            n_prune = 150 #input_features.shape[0] // 2 #50

            # img_logits_prune = logits_per_image[:, indices_txt[n_prune:]][indices_txt[n_prune:],:]
            # txt_logits_prune = logits_per_text[:, indices_img[n_prune:]][indices_img[n_prune:],:]
            cls_method = 3
            if cls_method == 1:
                img_logits_prune = logits_per_image[:, indices_txt[:-n_prune]][indices_txt[:-n_prune],:]
                txt_logits_prune = logits_per_text[:, indices_img[:-n_prune]][indices_img[:-n_prune],:]
            elif cls_method == 2:
                img_logits_prune = logits_per_image[:, indices_txt[n_prune:]][indices_txt[n_prune:],:]
                txt_logits_prune = logits_per_text[:, indices_img[n_prune:]][indices_img[n_prune:],:]
            else:
                img_logits_prune = (F.softmax(logits_per_image, dim=1) + 0.9 * F.softmax(logits_clstr_img, dim=1))/2
                txt_logits_prune = (F.softmax(logits_per_text, dim=1)  + 0.9 * F.softmax(logits_clstr_txt, dim=1))/2
                # img_logits_prune = (self.nsc._normalize_max(logits_per_image, axis=-1) + self.nsc._normalize_max(logits_clstr_img, axis=-1))/2
                # txt_logits_prune = (self.nsc._normalize_max(logits_per_text, axis=-1) + self.nsc._normalize_max(logits_clstr_txt, axis=-1))/2

                # img_logits_prune = (logits_per_image) # + logits_clstr_txt)/2           
                # txt_logits_prune = (logits_per_text)# + logits_clstr_txt)/2    

                # img_clstr_logits_prune = logits_clstr_img
                # txt_clstr_logits_prune = logits_clstr_txt  

            labels_prune = th.arange(txt_logits_prune.shape[0]).cuda()
            # labels_prune_img = labels[indices_txt[:-n_prune]]
            # labels_prune_txt = labels[indices_img[:-n_prune]]
            
            loss_i = self.criterion(img_logits_prune, labels_prune)
            loss_t = self.criterion(txt_logits_prune, labels_prune)
            # loss_ci = self.criterion(img_clstr_logits_prune, labels_prune)
            # loss_ct = self.criterion(txt_clstr_logits_prune, labels_prune)

            methow_w = 2
            w_temp = 0.009 #self.logit_scale #0.01
            z_w = 60
            if methow_w == 1:
                w_i =  ((avg_sim_img/sum(avg_sim_img)) / w_temp)
                w_t = ((avg_sim_txt/sum(avg_sim_txt)) / w_temp)
                # loss_i = loss_i * th.exp(w_i/ w_temp)
                # loss_t = loss_t * th.exp(w_t/ w_temp)
                loss_i = loss_i * w_i
                loss_t = loss_t * w_t
                loss_i = sum(loss_i) / (sum(w_i))
                loss_t = sum(loss_t) / (sum(w_t))
                # loss_i = sum(loss_i) / (logits_per_image.shape[0] * sum(th.exp(w_i/ w_temp)))
                # loss_t = sum(loss_t) / (logits_per_text.shape[0] * sum(th.exp(w_t/ w_temp)))
            elif methow_w == 2:
                w_i =  ((avg_sim_img/sum(avg_sim_img)))
                w_t = ((avg_sim_txt/sum(avg_sim_txt)))
                loss_i = loss_i * th.exp(w_i/ w_temp)
                loss_t = loss_t * th.exp(w_t/ w_temp)
                # loss_ci = loss_ci * th.exp(w_i/ w_temp)
                # loss_ct = loss_ct * th.exp(w_t/ w_temp)

                loss_i = sum(loss_i) / (sum(th.exp(w_i/ w_temp)))
                loss_t = sum(loss_t) / (sum(th.exp(w_t/ w_temp)))
                # loss_ci = sum(loss_ci) / (sum(th.exp(w_i/ w_temp)))
                # loss_ct = sum(loss_ct) / (sum(th.exp(w_t/ w_temp)))

            elif methow_w == 3:
                w_i =  ((avg_sim_img/sum(avg_sim_img)) ) * z_w
                w_t = ((avg_sim_txt/sum(avg_sim_txt)) ) * z_w
                # loss_i = loss_i * th.exp(w_i/ w_temp)
                # loss_t = loss_t * th.exp(w_t/ w_temp)
                loss_i = loss_i * w_i
                loss_t = loss_t * w_t
                loss_i = sum(loss_i) / (sum(w_i))
                loss_t = sum(loss_t) / (sum(w_t))
            else:
                w_i =  ((avg_sim_img/sum(avg_sim_img)))
                w_t = ((avg_sim_txt/sum(avg_sim_txt)))
                loss_i = loss_i * th.exp(w_i/ w_temp)
                loss_t = loss_t * th.exp(w_t/ w_temp)

                loss_i = sum(loss_i) / (logits_per_image.shape[0])
                loss_t = sum(loss_t) / (logits_per_text.shape[0])

        loss = ((loss_i + loss_t) / 2 )  + 0.5 * self.nsc(image_features, text_features) #self.nsc(logits_per_image, logits_per_text) #self.nsc(image_features, text_features)# + 0.2*(loss_ci + loss_ct)) / 4.

        # loss = (loss_i.mean() + loss_t.mean()) / 2.
        return loss

class CLIP(nn.Module):
    """
    Regular Contrastive Loss between 2 groups of embeddings
    """

    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(th.ones([]))
        # self.criterion = th.nn.CrossEntropyLoss(reduction="sum")
        self.criterion = th.nn.CrossEntropyLoss()
        self.temperature = 0.07 #0.04

    def forward(self, image_features, text_features):
        """
        Inputs shape (batch, embed_dim)

        Args:
            im: Visual embeddings (batch, embed_dim)
            s: Text embeddings (batch, embed_dim)

        Returns:
        """
        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        # logit_scale = self.logit_scale.exp()
        # logits_per_iamge = logit_scale * image_features @ text_features.t()
        # logits_per_text = logit_scale * text_features @ image_features.t()

        logits_per_iamge = image_features @ text_features.t()
        logits_per_text = text_features @ image_features.t()
        # logits_per_iamge /= (self.temperature * self.logit_scale.exp())
        # logits_per_text /= (self.temperature * self.logit_scale.exp())
        logits_per_iamge /= (self.temperature)
        logits_per_text /= (self.temperature)

        # logits = torch.matmul(img_feats, text_feats.T) * torch.exp(self.temperature_factor)

        labels = th.arange(image_features.shape[0]).cuda()
        # print(labels)

        loss_i = self.criterion(logits_per_iamge, labels)
        loss_t = self.criterion(logits_per_text, labels)

        loss = (loss_i.mean() + loss_t.mean()) / 2.
        return loss

class ContrastiveLoss_v1(nn.Module):
    """
    Regular Contrastive Loss between 2 groups of embeddings
    """

    def __init__(self, margin: float, max_violation: bool = False, norm: bool = True, use_cuda: bool = True):
        super().__init__()
        self.margin = margin
        self.sim = cosine_sim
        self.norm = norm
        self.max_violation = max_violation
        self.use_cuda = use_cuda

    def _pdist(self, A, B):
        prod = th.mm(A, B.t())
        norm = prod.diag().unsqueeze(1).expand_as(prod)
        res = (norm + norm.t() - 2 * prod).clamp(min = 0)
        return res.sqrt()

    def forward(self, im, s):
        """
        Inputs shape (batch, embed_dim)

        Args:
            im: Visual embeddings (batch, embed_dim)
            s: Text embeddings (batch, embed_dim)

        Returns:
        """
        # compute image-sentence score matrix - how close is im(y) to s(x)
        scores = self.sim(im, s)
        # distances = self._pdist(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals, where there is just the margin left
        mask: th.Tensor = th.eye(scores.shape[0]).bool()
        if self.use_cuda:
            mask = mask.cuda(non_blocking=True)
        cost_s = cost_s.masked_fill_(mask, 0)
        cost_im = cost_im.masked_fill_(mask, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        num_neg = 10
        neg_id = th.argsort(scores, dim=0, descending=True)
        cost_s_mask = cost_im[neg_id[:, 0:num_neg]].mean()

        cost_im_mask = []
        neg_id = th.argsort(scores, dim=1, descending=True)
        cost_im_mask = cost_s[neg_id[0:num_neg, :]].mean()


        # if self.norm:
        #     return (cost_s.sum() + cost_im.sum()).div(im.shape[0] * s.shape[0])
        return (cost_s_mask + cost_im_mask + cost_s.mean() + cost_im.mean()) / 2


def compute_mean_distance_l2(c, s):
    return th.mean((c - s) ** 2, dim=-1)


def compute_mean_distance_negative_l2(c, s):
    return -compute_mean_distance_l2(c, s)


class CycleConsistencyLoss(nn.Module):
    """
    Cycle Consistency Loss

    Default values are the resulted best
    """

    def __init__(self, num_samples: int = 1, compute_half_cycles: bool = False, use_cuda: bool = True,
                 verbose: bool = False, print_fn: Callable = print):
        super().__init__()
        self.compute_half_cycles = compute_half_cycles
        self.use_cuda = use_cuda
        self.print_fn = print_fn
        self.verbose = verbose
        self.num_samples = num_samples
        self.num_samples_tensor = (th.ones(1) * self.num_samples)
        if self.use_cuda:
            self.num_samples_tensor = self.num_samples_tensor.cuda(
                non_blocking=True)

        # define loss functions (currently L2)
        self.loss_distance_fn = compute_mean_distance_l2
        self.proximity_fn = compute_mean_distance_negative_l2
        self.proximity_mask_val = -INF

        self.softmax_temp = 1
        self.softmax = nn.Softmax(dim=-1)
        self.weight_index_simple = 1
        self.weight_index_gauss = 0
        self.lambda_index_gauss = 1
        self.var_denom_eps = 1e-8
        self.var_log_eps = 1

    def forward(self, clip_emb: th.FloatTensor, clip_mask: th.BoolTensor, clip_lens: th.LongTensor,
                sent_emb: th.FloatTensor, sent_mask: th.BoolTensor, sent_lens: th.LongTensor):
        """
        Args:
            clip_emb: (batch_size, num_clips, feat_dim)
            clip_mask: (batch_size, num_clips), False = real, True = masked
            clip_lens: (batch_size), corresponds to mask
            sent_emb: (batch_size, num_sents, feat_dim)
            sent_mask: (batch_size, num_sents), False = real, True = masked
            sent_lens: (batch_size), corresponds to mask

        Returns:
            CC clip loss, CC sentence loss
        """
        # Invert masks here s.t. padded sequence elements are 0
        clip_mask = ~clip_mask
        sent_mask = ~sent_mask

        # Get maximum of the sequence lengths
        clip_max_len = clip_mask.shape[1]
        sent_max_len = sent_mask.shape[1]

        # go from clips to sentences
        clip_sent_nn, clip_alpha, clip_alpha_raw = self.get_soft_nn(clip_emb, clip_mask, sent_emb, sent_mask)

        # calculate loss clips to sentences
        clip_sent_loss = None
        if self.compute_half_cycles:
            clip_sent_loss = self.get_total_loss(clip_emb, clip_sent_nn, clip_mask, clip_lens, clip_max_len, clip_alpha,
                                                 clip_alpha_raw)

        # go from those new sentences back to clips
        clip_clip_nn, clip_beta, clip_beta_raw = self.get_soft_nn(clip_sent_nn, clip_mask, clip_emb, clip_mask)

        # calculate loss on clip cycle consistency
        clip_clip_loss = self.get_total_loss(clip_emb, clip_clip_nn, clip_mask, clip_lens, clip_max_len, clip_beta,
                                             clip_beta_raw)

        # go from sentences to clips
        sent_clip_nn, sent_alpha, sent_alpha_raw = self.get_soft_nn(sent_emb, sent_mask, clip_emb, clip_mask)

        # calculate loss sentences to clips
        sent_clip_loss = None
        if self.compute_half_cycles:
            sent_clip_loss = self.get_total_loss(sent_emb, sent_clip_nn, sent_mask, sent_lens, sent_max_len, sent_alpha,
                                                 sent_alpha_raw)

        # go from those new clips back to sentences
        sent_sent_nn, sent_beta, sent_beta_raw = self.get_soft_nn(sent_clip_nn, sent_mask, sent_emb, sent_mask)

        # calculate loss on sentence cycle consistency
        sent_sent_loss = self.get_total_loss(sent_emb, sent_sent_nn, sent_mask, sent_lens, sent_max_len, sent_beta,
                                             sent_beta_raw)

        return clip_clip_loss, sent_sent_loss, clip_sent_loss, sent_clip_loss

    def get_mxn_repr(self, source_emb, source_mask, target_emb, target_mask):
        """
        Unsqueeze tensors and modify the mask accordingly to do N*M
        computations on sequence lengths N and M.
        Used to e.g. calculate distance between all source and all target
        embeddings.

        Args:
            source_emb: (batch_size, len_seq_source, feat_dim)
            source_mask: (batch_size, len_seq_source), 1 = real, 0 = masked
            target_emb: (batch_size, len_seq_target, feat_dim)
            target_mask: (batch_size, len_seq_target), 1 = real, 0 = masked

        Returns:
            source_rep, target_rep, total_mask
        """
        # unsqueeze source_emb in 2nd dimension
        source_rep = source_emb.unsqueeze(2)

        # unsqueeze target_emb in 1st dimension
        target_rep = target_emb.unsqueeze(1)

        # build mask that is 0 whenever either source or target mask is 0
        # only source mask is NOT enough for soft NN (tested)
        total_mask = source_mask.unsqueeze(2) & target_mask.unsqueeze(1)

        return source_rep, target_rep, total_mask

    def get_soft_nn(self, source_emb, source_mask, target_emb, target_mask):
        """
        Find soft nearest neighbors of each source_emb, looking for
        neighbors in target_emb.

        Args:
            source_emb: (batch_size, len_seq_source, feat_dim)
            source_mask: (batch_size, len_seq_source), 1 = real, 0 = masked
            target_emb: (batch_size, len_seq_target, feat_dim)
            target_mask: (batch_size, len_seq_target), 1 = real, 0 = masked

        Returns:
            soft_nn: (batch_size, len_seq_source) one nearest neighbor in the
                target space for each embedding in the source space
            weights: (batch_size, len_seq_source, len_seq_target) softmax
                output of similarity between each source and target pair.
                determines how much weight is given to each target embedding
                when calculating the nearest neighbor for a given source
                embedding.
            distance: (batch_size, len_seq_source, len_seq_target) unnormalized
                similarity weight (useful for e.g. crossentropyloss that
                expects unnormalized probabilities)
        """
        # get representation that allows to work on all
        # possible combinations of source and taret
        source_rep, target_rep, total_mask = self.get_mxn_repr(source_emb, source_mask, target_emb, target_mask)

        # calculate some distance on all combinations at once
        # in this case, negative L2 distance as measure of proximity
        distance = self.proximity_fn(source_rep, target_rep)
        # shape (batch_size, num_clips, num_clips)
        # d holds distances (batch_size, source_num, target_num)

        # set masked distances to (almost) negative infinity
        distance.masked_fill_(~total_mask, self.proximity_mask_val)
        # shape (batch_size, source_max_len, target_max_len)
        # masked values are set to very high negative number for softmax

        # calculate weights with softmax and some temperature
        # higher temp: uniform dist. lower temp: hard argmax
        weights_alpha = self.softmax(distance / self.softmax_temp)

        # with weights, calculate soft nearest neighbor in target
        # embedding space
        soft_nn = target_emb.unsqueeze(dim=1) * weights_alpha.unsqueeze(dim=3)
        soft_nn = th.sum(soft_nn, dim=2)

        return soft_nn, weights_alpha, distance

    # pylint: disable=unused-argument
    def get_total_loss(self, emb_orig, emb_nn, emb_mask, emb_lens, emb_max_len, beta, beta_raw):
        """
        Given embeddings and their cycled nearest neighbors,
        calculate total loss given the config flags

        Args:
            emb_orig: (batch_size, len_seq, feat_dim)
            emb_nn: (batch_size, len_seq, feat_dim)
            emb_mask: (batch_size, len_seq), 1 = real, 0 = masked
            emb_lens: (batch_size), corresponds to mask
            emb_max_len: int, th.max over emb lens dim -1
            beta: (batch_size, len_seq, len_seq) softmax weights
            beta_raw: (batch_size, len_seq, len_seq) similarity scores before
                softmax

        Returns:
            float loss
        """
        l_seq = th.zeros_like(emb_mask).float()
        batch_size, _ = emb_mask.shape
        if self.use_cuda:
            l_seq = l_seq.cuda(non_blocking=True)
        if self.weight_index_gauss != 0 or self.weight_index_simple != 0:
            (loss_simple_per_seq, loss_gauss_per_seq, var_reg_per_seq) = self.compute_loss_index_gauss(
                emb_mask, emb_lens, emb_max_len, beta)
            l_seq += (loss_gauss_per_seq + var_reg_per_seq) * self.weight_index_gauss
            l_seq += loss_simple_per_seq * self.weight_index_simple

        # subsample loss if requested
        if self.num_samples != -1:
            # check max amount of samples possible (depends on number of clips)
            n_samp = th.min(emb_lens, self.num_samples_tensor)
            # draw n_samp random integers without replacement in range emb_lens
            total_loss = 0
            for _batch, (c_loss, c_mask, c_nsamp) in enumerate(zip(l_seq, emb_mask, n_samp)):
                idx = th.multinomial(c_mask.float(), int(c_nsamp))
                total_loss += c_loss[idx].mean()
            total_loss /= batch_size
        else:
            # no subsampling, average over all losses
            total_loss = (l_seq.sum(dim=-1) / emb_lens).mean(dim=-1)

        return total_loss

    def compute_loss_index_gauss(self, emb_mask, _emb_lens, emb_max_len, beta):
        """
        Compute distance between original index and soft index.
        Takes into account variance between original and soft index.
        Also returns the version without variance.

        Returns total loss and loss per sequence / per batch, to be able
        to sample only some of the losses.

        Args:
            emb_mask: value mask (batch, seq_len), 1 = real value, 0 = masked
            _emb_lens: unused lengths of sequence, shape (batch)
            emb_max_len: th.max over emb_lens dim -1
            beta: softmax weight used to calculate the nearest neighbor
                (batch, seq_len, seq_len): dim 1 is the nearest neighbors in the
                sentence space the computation started, dim 2 is the original
                embeddings

        Returns:
            loss_gauss, loss_simple, loss_gauss_per_batch,
            loss_simple_per_batch, loss_gauss_per_seq, loss_simple_per_seq
        """
        # original index = arange
        idx_orig = th.arange(emb_max_len)
        if self.use_cuda:
            idx_orig = idx_orig.cuda(non_blocking=True)
        # add batch dim
        idx_orig.unsqueeze_(0)
        # shape (1, seq_len)

        # compute soft nearest neighbor index as sum of original indices
        # weighted by the softmax weights
        index_nn = th.sum(idx_orig.unsqueeze(1) * beta, dim=-1)
        # shape (batch, seq_len)

        # get mask and indices in correct representation
        idx_nn_rep, idx_orig_rep, emb_mask_rep = self.get_mxn_repr(index_nn, emb_mask, idx_orig, emb_mask)

        # get distance of each NN index to each original index
        # add an artificial dimension as feature dimension to make the same
        # math work out on indices that works on embeddings
        distance = self.loss_distance_fn(idx_nn_rep.unsqueeze(-1), idx_orig_rep.unsqueeze(-1))
        # shape (batch, seq_len, seq_len)

        # mask values that exceed the sequence length
        distance.masked_fill_(~emb_mask_rep, 0)

        # diagonal of last 2 dims of this distance tensor contains distance
        # from soft index i to hard index i, this is the loss distance
        loss_simple_per_seq = distance.diagonal(dim1=-2, dim2=-1)
        # shape (batch, seq_len)

        # to get variance, multiply with beta (softmax output weights)
        # and sum over the row
        variance = th.sum(distance * beta, dim=-1)

        # calculate regularizer loss on the variance and apply mask
        var_reg_per_seq = self.lambda_index_gauss * .5 * th.log(self.var_log_eps + variance)
        var_reg_per_seq.masked_fill_(emb_mask, 0)
        # shape (batch, seq_len)

        # calculate loss (no need to apply mask since distance is masked to 0)
        loss_gauss_per_seq = loss_simple_per_seq / (variance + self.var_denom_eps) + var_reg_per_seq

        # for now return all the losses, in case we want to sample some
        # sequences only
        return loss_simple_per_seq, loss_gauss_per_seq, var_reg_per_seq



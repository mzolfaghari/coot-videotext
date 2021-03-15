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

class CoCLR(th.nn.Module):
    def __init__(self):
        super(CoCLR, self).__init__()

        self.K = 3000
        dim = 384 #network embedding size
        dim_org_vid = 512
        dim_org_txt = 1536
        self.T=0.05
        self.topk=5


        # create the queue
        self.register_buffer("queue", th.randn(dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", th.zeros(1, dtype=th.long))

        # create another queue, for the second view of the data
        self.register_buffer("queue_second", th.randn(dim_org_txt, self.K))
        self.queue_second = nn.functional.normalize(self.queue_second, dim=0)
        
        # for monitoring purpose only
        self.register_buffer("queue_label", th.ones(self.K, dtype=th.long) * -1)
        
        self.queue_is_full = False

    @th.no_grad()
    def _dequeue_and_enqueue(self, keys, keys_second):

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        # assert self.K % batch_size == 0  # for simplicity


        ## ===============================================
        ptr_end = ptr+batch_size
        ptr_end_k = batch_size
        if (ptr+batch_size) > self.K:
            ptr_end = self.K
            ptr_end_k = self.K - (ptr + batch_size)
        # print(ptr_end, ptr_end_k)
        # import pdb; pdb.set_trace()
        self.queue[:, ptr:ptr_end] = keys.T[:, :ptr_end_k]
        self.queue_second[:, ptr:ptr_end] = keys_second.T[:, :ptr_end_k]
        self.queue_label[ptr:ptr_end] = th.ones_like(keys[:ptr_end_k,0])


        ## ===============================================
        # # replace the keys at ptr (dequeue and enqueue)
        # self.queue[:, ptr:ptr + batch_size] = keys.T
        # self.queue_second[:, ptr:ptr + batch_size] = keys_second.T
        # self.queue_vname[ptr:ptr + batch_size] = vnames
        # self.queue_label[ptr:ptr + batch_size] = torch.ones_like(keys[:,0])

        if (ptr+batch_size) > self.K:
            ptr = 0
        else:
            ptr = (ptr + batch_size) #% self.K  # move pointer

        self.queue_ptr[0] = ptr
    
    
    def compute_loss(self, logits, mask):
        mask_sum = mask.sum(1)
        loss = - th.log( (F.softmax(logits, dim=1) * mask).sum(1) )
        return loss.mean()

    def forward(self, video_embd, text_embd, keys_vid_org, keys_txt_org):

        keys_txt_kf = th.mean(keys_txt_org, dim=1)

        kf = nn.functional.normalize(keys_txt_kf, dim=1)
        q = nn.functional.normalize(video_embd, dim=1)
        k = nn.functional.normalize(text_embd, dim=1)
        # compute logits
        l_pos1 = th.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_pos2 = th.einsum('nc,nc->n', [k, q]).unsqueeze(-1)
        l_neg1 = q @ q.t()
        l_neg2 = k @ k.t()
        # l_neg = th.einsum('nc,ck->nk', [q, self.queue.clone().detach().cuda()])

        # logits: N,(1+K)
        logits = th.cat([l_pos1, l_pos2, l_neg1, l_neg2], dim=1)

        # apply temperature
        logits /= self.T

        mask1 = th.zeros(kf.shape[0], kf.shape[0]).bool()
        mask2 = th.zeros(kf.shape[0], kf.shape[0]).bool()
        # if not self.queue_is_full:
        #     self.queue_is_full = th.all(self.queue_label != -1)
        #     if self.queue_is_full: print('\n===== queue is full now =====')

        # if self.queue_is_full and (self.topk != 0):
        mask_sim = kf @ kf.t() #self.queue_second.clone().detach().cuda())
        # mask_sim[mask_source] = - np.inf # mask out self (and sibling videos)
        _, topkidx = th.topk(mask_sim, self.topk, dim=1)
        topk_onehot = th.zeros_like(mask_sim)
        topk_onehot.scatter_(1, topkidx, 1)
        mask1[topk_onehot.bool()] = True
        mask2[topk_onehot.bool()] = True
        
        mask = th.cat([mask1, mask2], dim=1)
        mask = th.cat([th.ones((mask.shape[0],1), dtype=th.long, device=mask.device).bool(),
                          mask], dim=1)
        mask = th.cat([th.ones((mask.shape[0],1), dtype=th.long, device=mask.device).bool(),
                          mask], dim=1)
        self._dequeue_and_enqueue(k, keys_txt_kf)
        # import pdb; pdb.set_trace()
        return self.compute_loss(logits, mask.detach().cuda()) 

class MILNCELoss(th.nn.Module):
    def __init__(self):
        super(MILNCELoss, self).__init__()

    def forward(self, video_embd, text_embd, x=0, y=0):
        video_embd = nn.functional.normalize(video_embd, dim=1)
        text_embd = nn.functional.normalize(text_embd, dim=1)
        x = th.matmul(video_embd, text_embd.t()) / 0.05
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
        temperature = 0.3
        tau_plus = 0.01
        debiased = 1
       # neg score
        video_embd = nn.functional.normalize(video_embd, dim=1)
        text_embd = nn.functional.normalize(text_embd, dim=1)

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
    CrossCLR Loss between 2 groups of embeddings
    """

    def __init__(self, temperature=0.05, logger = None):
        super().__init__()
        self.logit_scale = nn.Parameter(th.ones([]))
        # self.criterion = th.nn.CrossEntropyLoss(reduction="sum")
        self.criterion = th.nn.CrossEntropyLoss(reduction='none') #th.nn.CrossEntropyLoss()
        self.temperature = temperature #0.04
        self.logger = logger
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

    def compute_loss(self, logits, mask):
        mask_sum = mask.sum(1)
        loss = - th.log( (F.softmax(logits, dim=1) * mask).sum(1) )
        return loss #loss.mean()

    def forward(self, image_features, text_features, input_vid, input_txt):
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
        thrshold = 0.75#0.8
        cls_method = 6

        # =====
        ## Tomorrow test with removing lines 412 and 414! Normalization might remove some info!
        # also have look on normaliztion line 709, 710
        #======
        # thrsh_vid = 0.8 #0.012 #threshold for image similarity - find influential samples
        # thrsh_txt = 170 #threshold for text similarity - find influential samples


        use_emb = 0 # Use network embeddings or original embeddings to weight loss
        w_temp = 0.0035 # Temperature for scaling weights. self.logit_scale # Naming was wrong 1e1 means 1e2!!
        prune_percent = 0.25 # percentage of pruning negatives ! This will be used only for no_prune=2
        # In some of experiments I wrongly used prune percent instead of weight negative
        w_negative = 0.8 #0.8 weight of negative scores.


        input_vid1 = th.mean(input_vid, dim=1)
        input_txt1 = th.mean(input_txt, dim=1)
        # This a new normalization for input features! We didn't have for experiments < 9 Feb 2021
        input_vid2 = input_vid1 #/ input_vid1.max(dim=-1, keepdim=True)[0]
        # import pdb; pdb.set_trace()
        input_txt2 = input_txt1 #/ input_txt1.max(dim=-1, keepdim=True)[0]
        # input_vid2 = input_txt2
        # input_vid = input_txt
        # import pdb; pdb.set_trace()

        # ===================================================
        # normalized features
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = nn.functional.normalize(image_features, dim=1)
        text_features = nn.functional.normalize(text_features, dim=1)

        self._dequeue_and_enqueue(image_features, text_features, input_vid2, input_txt2)

        logits_per_image = image_features @ text_features.t()
        logits_per_text = text_features @ image_features.t()

        logits_clstr_vid = image_features @ image_features.t()
        logits_clstr_txt = text_features @ text_features.t()

        # logits_clstr_vid = image_features @ self.queue_vid.cuda()
        # logits_clstr_txt = text_features @ self.queue_txt.cuda()

        logits_per_image /= self.temperature #* self.logit_scale.exp()
        logits_per_text /= self.temperature #* self.logit_scale.exp()
        logits_clstr_vid /= self.temperature #* self.logit_scale.exp()
        logits_clstr_txt /= self.temperature #* self.logit_scale.exp()

        qptr_end = self.queue_ptr[0]+image_features.shape[0]
        if (self.queue_ptr[0]+image_features.shape[0]) > self.K:
            qptr_end = self.K
            inp_feat_k = image_features.shape[0] - (self.queue_ptr[0]+image_features.shape[0] - self.K)
            positive_mask =  self._get_positive_mask_bank(self.K, image_features.shape[0], self.queue_ptr[0])
        else:
            positive_mask =  self._get_positive_mask_bank(self.K, image_features.shape[0], self.queue_ptr[0]) #self._get_positive_mask(self.K)[self.queue_ptr[0]:qptr_end,:]
            
        sim_scores_vid = (input_vid2 @ self.queue_vid_org.cuda()) * positive_mask
        sim_scores_txt= (input_txt2 @ self.queue_txt_org.cuda()) * positive_mask
            

        
        if weighted_loss == 0:

            positive_mask =  self._get_positive_mask(image_features.shape[0]).type(th.bool)
            logits_clstr_vid = logits_clstr_vid * positive_mask
            logits_clstr_txt = logits_clstr_txt * positive_mask
            labels = th.arange(logits_per_image.shape[0]).cuda()
            

            if intra_modality:
                if no_softmax:  
                    vid_logits_prune = th.cat([logits_per_image, w_negative * logits_clstr_vid], dim=1)
                    txt_logits_prune = th.cat([logits_per_text, w_negative * logits_clstr_txt], dim=1)
                else: 
                    vid_logits_prune = th.cat([F.softmax(logits_per_image, dim=1), w_negative * F.softmax(logits_clstr_vid, dim=1)], dim=1)
                    txt_logits_prune = th.cat([F.softmax(logits_per_text, dim=1), w_negative * F.softmax(logits_clstr_txt, dim=1)], dim=1)
            else:
                if no_softmax:  
                    vid_logits_prune = logits_per_image
                    txt_logits_prune = logits_per_text
                else: 
                    vid_logits_prune = F.softmax(logits_per_image, dim=1)
                    txt_logits_prune = F.softmax(logits_per_text, dim=1)

            loss_i = self.criterion(vid_logits_prune, labels)
            loss_t = self.criterion(txt_logits_prune, labels)
            loss_i = loss_i.mean()
            loss_t = loss_t.mean()

        elif weighted_loss == 1:

            if use_emb:
                input_vid = image_features
                input_txt = text_features
            avg_sim_vid = th.mean(sim_scores_vid,dim=1)
            avg_sim_txt = th.mean(sim_scores_txt,dim=1)
            

            sorted_vid, indices_vid = th.sort(avg_sim_vid)
            sorted_txt, indices_txt = th.sort(avg_sim_txt)
            sorted_vid = sorted_vid / sorted_vid.max(dim=-1, keepdim=True)[0]
            sorted_txt = sorted_txt / sorted_txt.max(dim=-1, keepdim=True)[0]


            # ======================================================
            # Find index of influential samples and remove them from negative set
            indices_vid_thrsh = indices_vid[sorted_vid<thrshold]
            indices_txt_thrsh = indices_txt[sorted_txt<thrshold]
            # indices_vid_thrsh = indices_txt_thrsh

            # # import pdb; pdb.set_trace()
            # print("Image:", sorted_vid[1],sorted_vid[-1])  #0.012
            # print("Text:", sorted_txt[1],sorted_txt[-1])   #90
            # print("==="*20)    
            labels = th.arange(image_features.shape[0]).cuda()
            # print(logits_per_image.shape)

            # print(labels)
            # vid_logits_prune = logits_per_image[:, indices_txt[n_prune:]][indices_txt[n_prune:],:]
            # txt_logits_prune = logits_per_text[:, indices_vid[n_prune:]][indices_vid[n_prune:],:]
            if cls_method == 6:
                # --------- loss_CrossCLR_RemClstrPos_Prune -----
                # n_prune = int((logits_clstr_vid.shape[0]  - prune_percent *  logits_clstr_vid.shape[0]) // 1)
                # n_prune = (logits_clstr_vid.shape[0] - 200
                positive_mask =  self._get_positive_mask(logits_clstr_vid.shape[0]).type(th.bool)
                # positive_mask =  self._get_positive_mask(self.K)[self.queue_ptr[0]:self.queue_ptr[0]+logits_clstr_vid.shape[0],:].type(th.bool)
                logits_clstr_vid = logits_clstr_vid * positive_mask
                logits_clstr_txt = logits_clstr_txt * positive_mask

                if no_prune == 0:
                    negatives_vid = logits_clstr_vid
                    negatives_txt = logits_clstr_txt

                elif no_prune == 1:

                    negatives_vid = logits_clstr_vid[:, indices_vid_thrsh]
                    negatives_txt = logits_clstr_txt[:, indices_txt_thrsh]

                else:
                    negatives_vid = logits_clstr_vid[:, indices_vid[:n_prune]]
                    negatives_txt = logits_clstr_txt[:, indices_txt[:n_prune]]


                if intra_modality:
                    vid_logits_prune = th.cat([logits_per_image, w_negative * negatives_vid], dim=1)
                    txt_logits_prune = th.cat([logits_per_text, w_negative * negatives_txt], dim=1)

                else:
                    vid_logits_prune = logits_per_image
                    txt_logits_prune = logits_per_text

                
                labels_prune = th.arange(logits_per_image.shape[0]).cuda()
                # diag = np.eye(logits_per_image.shape[0])
                # mask = th.from_numpy((diag)).cuda()
                # mask_neg_v = th.zeros_like(negatives_vid)
                # mask_neg_t = th.zeros_like(negatives_txt)
                # mask_v = th.cat([mask, mask_neg_v], dim=1)
                # mask_t = th.cat([mask, mask_neg_t], dim=1)
                # import pdb;pdb.set_trace()
                # loss_i = self.compute_loss(vid_logits_prune, mask_v)
                # loss_t = self.compute_loss(txt_logits_prune, mask_t)

                loss_i = self.criterion(vid_logits_prune, labels_prune)
                loss_t = self.criterion(txt_logits_prune, labels_prune)

            else:
                # --------- loss_CrossCLR_RemClstrPos_Prune -----
                n_prune = int((logits_clstr_vid.shape[0]  - prune_percent *  logits_clstr_vid.shape[0]) // 1)
                # n_prune = (logits_clstr_vid.shape[0] - 200
                positive_mask =  self._get_positive_mask(logits_clstr_vid.shape[0]).type(th.bool)
                logits_clstr_vid = logits_clstr_vid * positive_mask
                logits_clstr_txt = logits_clstr_txt * positive_mask

                if no_prune == 0:
                    negatives_vid = logits_clstr_vid
                    negatives_txt = logits_clstr_txt

                elif no_prune == 1:
                    negatives_vid = logits_clstr_vid[:, indices_vid_thrsh]
                    negatives_txt = logits_clstr_txt[:, indices_vid_thrsh]                    
                else:
                    negatives_vid = logits_clstr_vid[:, indices_vid[:n_prune]]
                    negatives_txt = logits_clstr_txt[:, indices_txt[:n_prune]]

                if intra_modality:
                    vid_logits_prune1 = logits_per_image
                    vid_logits_prune2 = th.cat([th.diag(logits_per_image).view(logits_per_image.shape[0], 1), w_negative * negatives_vid], dim=1)
                    txt_logits_prune1 = logits_per_text
                    txt_logits_prune2 = th.cat([th.diag(logits_per_text).view(logits_per_image.shape[0], 1), w_negative * negatives_txt], dim=1)
            
                    labels_prune1 = th.arange(logits_per_image.shape[0]).cuda()
                    labels_prune2 = th.zeros(logits_per_image.shape[0]).cuda(non_blocking=True).long()
                    
                    loss_i = (self.criterion(vid_logits_prune1, labels_prune1) + self.criterion(vid_logits_prune2, labels_prune2))
                    loss_t = (self.criterion(txt_logits_prune1, labels_prune1) +  self.criterion(txt_logits_prune2, labels_prune2))
                else:
                    vid_logits_prune = logits_per_image
                    txt_logits_prune = logits_per_text  
                    labels_prune = th.arange(logits_per_image.shape[0]).cuda()
                    loss_i = self.criterion(vid_logits_prune, labels_prune)
                    loss_t = self.criterion(txt_logits_prune, labels_prune)   



            w_i =  ((avg_sim_vid/sum(avg_sim_vid)))
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

        # logits = torch.matmul(vid_feats, text_feats.T) * torch.exp(self.temperature_factor)

        labels = th.arange(image_features.shape[0]).cuda()
        # print(labels)

        loss_i = self.criterion(logits_per_iamge, labels)
        loss_t = self.criterion(logits_per_text, labels)

        loss = (loss_i.mean() + loss_t.mean()) / 2.
        return loss

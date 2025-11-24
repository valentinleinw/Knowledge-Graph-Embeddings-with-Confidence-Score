import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class TransEUncertainty(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TransEUncertainty, self).__init__()
        self.embedding_dim = embedding_dim
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
    
    # normal TransE scoring function
    def forward(self, h, r, t):
        distance = torch.norm(
        self.entity_embeddings(h) + self.relation_embeddings(r) - self.entity_embeddings(t),
        p=1, dim=1
    )
        scaled = distance / self.embedding_dim
        return torch.sigmoid(-scaled)  
    
    # TransE scoring function changed to Loss Function by using confidence scores
    """def loss(self, pos_triples, neg_triples, confidence_scores, margin=1.0):
        pos_loss = torch.sum(confidence_scores * torch.clamp(
            margin + self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2]) -
            self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2]), min=0))
        return torch.mean(pos_loss)
    """
    def loss(self, pred, conf):
        return torch.mean(torch.abs(pred - conf))  # MAE loss

    
    # TransE scoring function changed to Loss Function by using confidence scores and including negative confidence scores
    def loss_neg(self, pos_triples, neg_triples, pos_confidence_scores, neg_confidence_scores, margin=1.0):

        # Compute positive and negative scores
        pos_scores = self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
        neg_scores = self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2])
        
        num_neg_samples = len(neg_scores) // len(pos_scores)  # Get ratio of neg to pos
        pos_scores = pos_scores.repeat_interleave(num_neg_samples)  # Expand to match neg_scores
        pos_confidence_scores = pos_confidence_scores.repeat_interleave(num_neg_samples)

        # Compute loss with confidence weighting
        pos_loss = torch.sum(pos_confidence_scores * torch.clamp(margin + pos_scores - neg_scores, min=0))
        neg_loss = torch.sum(neg_confidence_scores * torch.clamp(margin + pos_scores - neg_scores, min=0)) 

        total_loss = pos_loss + neg_loss
        return torch.mean(total_loss)
        
    def objective_function(self, pos_triples, neg_triples, confidence_scores):

        # Compute the scores for positive and negative triples
        pos_scores = torch.sigmoid(-self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2]))
        neg_scores = torch.sigmoid(-self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2]))

        # First term: MSE loss for positive triples (f(l) - s_l)^2
        loss_pos = torch.mean((pos_scores - confidence_scores) ** 2)

        margin = 0.5
        loss_neg = torch.mean(F.relu(neg_scores - margin) ** 2)

        # Total objective function
        return loss_pos + loss_neg
    
    def softplus_loss(self, pos_pred, confidence_scores, neg_pred):
        
        
        loss_pos = torch.mean(confidence_scores * F.softplus(pos_pred))
        loss_neg = torch.mean(F.softplus(-neg_pred))
        return loss_pos + loss_neg
    
    def gaussian_nll_loss(self, pred, confidence_scores):

        sigma2 = (F.softplus(pred) + 1/(2 * math.pi))** 2
        #pos_scores = self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
        loss = torch.mean(0.5 *  torch.log(sigma2) + (pred - confidence_scores) ** 2 / (2 * sigma2))
        return loss
    
    def kl_divergence_loss(self, pos_pred, confidence_scores):

        p = F.softmax(pos_pred, dim=0)
        q = F.softmax(confidence_scores, dim=0)

        return F.kl_div(p.log(), q, reduction='batchmean')
   
import torch
import torch.nn as nn

class DistMultUncertainty(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(DistMultUncertainty, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        self.relation_log_sigma2 = nn.Embedding(num_relations, 1)

        # Minimum variance to guarantee NLL ≥ 0
        self.sigma2_min = 1.0 / (2 * math.pi)

    
    def forward(self, h, r, t):
        head_embedding = self.entity_embeddings(h)
        relation_embedding = self.relation_embeddings(r)
        tail_embedding = self.entity_embeddings(t)
        return torch.sum(head_embedding * relation_embedding * tail_embedding, dim=1)

    # MAE loss
    def loss(self, pred, conf):
        return torch.mean(torch.abs(pred - conf))

    """
    def loss(self, pos_triples, neg_triples, confidence_scores, margin=1.0):
        pos_score = self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
        neg_score = self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2])
        pos_loss = confidence_scores * F.relu(margin + pos_score - neg_score)
        return torch.mean(pos_loss)
    """ 
    def loss_neg(self, pos_triples, neg_triples, pos_confidence_scores, neg_confidence_scores, margin=1.0):
        pos_scores = -self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
        neg_scores = -self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2])

        num_neg_samples = len(neg_scores) // len(pos_scores)
        pos_scores = pos_scores.repeat_interleave(num_neg_samples)
        pos_confidence_scores = pos_confidence_scores.repeat_interleave(num_neg_samples)

        loss = pos_confidence_scores * F.relu(margin + pos_scores - neg_scores) \
                + neg_confidence_scores * F.relu(margin + pos_scores - neg_scores)
        return torch.mean(loss)
  
    def objective_function(self, pos_triples, neg_triples, confidence_scores):
        pos_scores = torch.sigmoid(self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2]))
        neg_scores = torch.sigmoid(self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2]))

        
        loss_pos = torch.mean((pos_scores - confidence_scores) ** 2)
        
        margin = 0.5
        loss_neg = torch.mean(F.relu(neg_scores - margin) ** 2)

        return loss_pos + loss_neg
    
    def softplus_loss(self, pos_pred, confidence_scores, neg_pred):
        
        pos_loss = torch.mean(-confidence_scores * F.logsigmoid(pos_pred))
        neg_loss = torch.mean(-F.logsigmoid(-neg_pred))

        return pos_loss + neg_loss

    def gaussian_nll_loss(self, h, r, t, confidence_scores):
        
        pred = self.forward(h,r,t)
        
        log_sigma2_raw = self.relation_log_sigma2(r).squeeze()

        # sigma^2 = softplus(raw) + sigma2_min
        sigma2 = F.softplus(log_sigma2_raw) + self.sigma2_min
        #pos_scores = self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
                
        return torch.mean(0.5 * torch.log(2 * math.pi * sigma2) +((pred - confidence_scores) ** 2) / (sigma2))

    def kl_divergence_loss(self, pos_pred, confidence_scores):

        p = F.softmax(pos_pred, dim=0)
        q = F.softmax(confidence_scores, dim=0)

        return F.kl_div(p.log(), q, reduction='batchmean')
    
import torch
import torch.nn as nn

class ComplExUncertainty(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Embeddings
        self.entity_re_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.entity_im_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_re_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.relation_im_embeddings = nn.Embedding(num_relations, embedding_dim)

        # Regression head
        self.relation_log_sigma2 = nn.Embedding(num_relations, 1)

        # minimum variance to ensure NLL >= 0
        self.sigma2_min = 1.0 / (2 * math.pi)


    def forward(self, h, r, t):
        head_real, head_imag = self.entity_re_embeddings(h), self.entity_im_embeddings(h)
        relation_real, relation_imag = self.relation_re_embeddings(r), self.relation_im_embeddings(r)
        tail_real, tail_imag = self.entity_re_embeddings(t), self.entity_im_embeddings(t)

        return torch.sigmoid(torch.sum(
            head_real * relation_real * tail_real + head_imag * relation_real * tail_imag + head_real * relation_imag * tail_imag - head_imag * relation_imag * tail_real,
            dim=1
        ))

    def loss(self, pred, conf):
        conf = conf.clamp(0.05, 0.95)
        return torch.mean(torch.abs(pred - conf))

    """
    def loss(self, pos_triples, neg_triples, confidence_scores, margin=1.0):
        pos_score = self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
        neg_score = self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2])

        loss = confidence_scores * F.relu(margin + neg_score - pos_score)
        return loss.mean()
"""
    
    def loss_neg(self, pos_triples, neg_triples, pos_confidence_scores, neg_confidence_scores, margin=1.0):
        pos_scores = self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
        neg_scores = self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2])

        num_neg_samples = len(neg_scores) // len(pos_scores)
        pos_scores = pos_scores.repeat_interleave(num_neg_samples)
        pos_confidence_scores = pos_confidence_scores.repeat_interleave(num_neg_samples)

        pos_loss = pos_confidence_scores * F.relu(margin + neg_scores - pos_scores)
        neg_loss = neg_confidence_scores * F.relu(margin + neg_scores - pos_scores)
        return (pos_loss.mean() + neg_loss.mean())
    
    def objective_function(self, pos_triples, neg_triples, confidence_scores):
        pos_scores = torch.sigmoid(self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2]))
        neg_scores = torch.sigmoid(self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2]))

        # Loss on positive triples (weighted MSE)
        loss_pos = torch.mean((pos_scores - confidence_scores) ** 2)

        margin = 0.5
        loss_neg = torch.mean((neg_scores - margin) ** 2)

        return loss_pos + loss_neg
    
    def softplus_loss(self, pos_pred, confidence_scores, neg_pred):
        
        pos_loss = torch.mean(-confidence_scores * F.logsigmoid(pos_pred))
        neg_loss = torch.mean(-F.logsigmoid(-neg_pred))

        return pos_loss + neg_loss
    
    def gaussian_nll_loss(self, h,r,t, confidence_scores):
        #pos_scores = self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])

        pred = self.forward(h, r, t)  # ComplEx real score

        # predicted log σ² for each relation
        log_sigma2_raw = self.relation_log_sigma2(r).squeeze(-1)

        # enforce σ² >= sigma2_min to avoid negative NLL
        sigma2 = F.softplus(log_sigma2_raw) + self.sigma2_min

        loss = 0.5 * ((pred - confidence_scores) ** 2 / (sigma2) + torch.log(2 * math.pi * sigma2))

        return loss.mean()
    
    
    def kl_divergence_loss(self, pos_pred, confidence_scores):

        p = F.softmax(pos_pred, dim=0)
        q = F.softmax(confidence_scores, dim=0)

        return F.kl_div(p.log(), q, reduction='batchmean')
    
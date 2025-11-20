import torch.nn as nn
import torch
import torch.nn.functional as F


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
    
    def softplus_loss(self, pos_triples, neg_triples, confidence_scores):
        
        pos_scores = self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
        neg_scores = self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2])
        
        
        loss_pos = torch.mean(confidence_scores * F.softplus(pos_scores))
        loss_neg = torch.mean(F.softplus(-neg_scores))
        return loss_pos + loss_neg
    
    def gaussian_nll_loss(self, pos_triples, confidence_scores):

        pos_scores = self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
        loss = torch.mean(0.5 *  torch.log(confidence_scores + 1e-8) + (pos_scores - confidence_scores) ** 2 / (2 * confidence_scores + 1e-8))
        return loss
    
    def kl_divergence_loss(self, pos_triples, confidence_scores):
        
        pos_scores = self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
        pos_probs = F.softmax(-pos_scores, dim=0)
        target_probs = F.softmax(-confidence_scores, dim=0)
        return F.kl_div(pos_probs.log(), target_probs, reduction='batchmean')
   
import torch
import torch.nn as nn

class DistMultUncertainty(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(DistMultUncertainty, self).__init__()
        self.embedding_dim = embedding_dim
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # Regression head to map raw DistMult score → confidence
        self.regressor = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output in [0,1]
        )

    def forward(self, h, r, t):
        h_emb = self.entity_embeddings(h)
        r_emb = self.relation_embeddings(r)
        t_emb = self.entity_embeddings(t)

        # Original DistMult score (scalar per triple)
        score = torch.sum(h_emb * r_emb * t_emb, dim=1, keepdim=True)  # Shape [batch,1]

        # Predict confidence
        score = score / self.embedding_dim  # now roughly in [-1,1]
        conf_pred = self.regressor(score)
        return conf_pred.squeeze(1)  # Shape [batch]

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
    
    def softplus_loss(self, pos_triples, neg_triples, confidence_scores):
        pos_scores = self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
        neg_scores = self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2])

        pos_loss = torch.mean(-confidence_scores * F.logsigmoid(pos_scores))
        neg_loss = torch.mean(-F.logsigmoid(-neg_scores))

        return pos_loss + neg_loss

    def gaussian_nll_loss(self, pos_triples, confidence_scores):
        
        pos_scores = self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
                
        return torch.mean(0.5 * torch.log(confidence_scores + 1e-8) +((pos_scores - confidence_scores) ** 2) / (2 * confidence_scores + 1e-8))

    def kl_divergence_loss(self, pos_triples, confidence_scores):
        pos_scores = self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])

        p = F.softmax(pos_scores, dim=0)
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
        self.regressor = nn.Sequential(
            nn.Linear(6 * embedding_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Small initial alpha
        self.alpha = nn.Parameter(torch.tensor(0.1))

        # ---- Initialization ----
        for emb in [self.entity_re_embeddings, self.entity_im_embeddings,
                    self.relation_re_embeddings, self.relation_im_embeddings]:
            nn.init.xavier_uniform_(emb.weight, gain=0.5)

        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, h, r, t):
        h_re, h_im = self.entity_re_embeddings(h), self.entity_im_embeddings(h)
        r_re, r_im = self.relation_re_embeddings(r), self.relation_im_embeddings(r)
        t_re, t_im = self.entity_re_embeddings(t), self.entity_im_embeddings(t)

        # ComplEx raw score
        score = torch.sum(
            h_re * r_re * t_re + h_im * r_re * t_im + h_re * r_im * t_im - h_im * r_im * t_re,
            dim=1,
            keepdim=True
        )

        # Scale and clamp raw score
        score_scaled = torch.clamp(self.alpha * score / self.embedding_dim, -10, 10)

        # Concatenate embeddings + score
        x = torch.cat([h_re, h_im, r_re, r_im, t_re, t_im, score_scaled], dim=1)

        # Predict confidence, clamp to [0,1]
        conf_pred = torch.clamp(self.regressor(x), 0.0, 1.0)
        return conf_pred.squeeze(1)

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
    
    def softplus_loss(self, pos_triples, neg_triples, confidence_scores):
        pos_score = self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
        neg_score = self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2])

        pos_loss = torch.mean(-confidence_scores * F.logsigmoid(pos_score + 1e-8))
        neg_loss = torch.mean(-F.logsigmoid(-neg_score - 1e-8))

        return pos_loss + neg_loss
    
    def gaussian_nll_loss(self, pos_triples, confidence_scores):
        pos_scores = self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])

        loss = 0.5 * ((pos_scores - confidence_scores) ** 2 / (2 * confidence_scores + 1e-8) + torch.log(confidence_scores + 1e-8))

        return loss.mean()
    
    
    def kl_divergence_loss(self, pos_triples, confidence_scores):
        pos_scores = self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
        
        p = F.softmax(pos_scores + 1e-8, dim=0)  # Model’s predicted distribution
        q = F.softmax(confidence_scores, dim=0)  # Confidence score as target

        return F.kl_div(p.log(), q, reduction="batchmean")

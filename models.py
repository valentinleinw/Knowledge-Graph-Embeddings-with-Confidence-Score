import torch.nn as nn
import torch
import torch.nn.functional as F


class TransEUncertainty(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TransEUncertainty, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
    
    # normal TransE scoring function
    def forward(self, h, r, t):
        return self.entity_embeddings(h) + self.relation_embeddings(r) - self.entity_embeddings(t)
    
    # TransE scoring function changed to Loss Function by using confidence scores
    def loss(self, pos_triples, neg_triples, confidence_scores, margin=1.0):
        pos_loss = torch.sum(confidence_scores * torch.clamp(
            margin + torch.norm(self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2]), p=1, dim=1) -
            torch.norm(self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2]), p=1, dim=1), min=0))
        return pos_loss
    
    
    def loss_neg(self, pos_triples, neg_triples, pos_confidence_scores, neg_confidence_scores, margin=1.0):

        # Compute positive and negative scores
        pos_scores = torch.norm(self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2]), p=1, dim=1)
        neg_scores = torch.norm(self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2]), p=1, dim=1)
        
        num_neg_samples = len(neg_scores) // len(pos_scores)  # Get ratio of neg to pos
        pos_scores = pos_scores.repeat_interleave(num_neg_samples)  # Expand to match neg_scores
        pos_confidence_scores = pos_confidence_scores.repeat_interleave(num_neg_samples)

        # Compute loss with confidence weighting
        pos_loss = torch.sum(pos_confidence_scores * torch.clamp(margin + pos_scores - neg_scores, min=0))
        neg_loss = torch.sum(neg_confidence_scores * torch.clamp(margin + pos_scores - neg_scores, min=0)) 

        total_loss = pos_loss + neg_loss
        return total_loss
    
    # replace relu with max(0,s_l−γ)^2
    
    def objective_function(self, pos_triples, neg_triples, confidence_scores):

        # Compute the scores for positive and negative triples
        pos_scores = torch.sigmoid(-torch.norm(self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2]), p=1, dim=1))
        neg_scores = torch.sigmoid(-torch.norm(self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2]), p=1, dim=1))

        # First term: MSE loss for positive triples (f(l) - s_l)^2
        loss_pos = torch.mean((pos_scores - confidence_scores) ** 2)

        margin = 0.5
        loss_neg = torch.mean(F.relu(neg_scores - margin) ** 2)

        # Total objective function
        return loss_pos + loss_neg
    
    def softplus_loss(self, pos_triples, neg_triples, confidence_scores):
        """
         Calculates the loss as:
         L = 𝔼_pos[ log(1 + exp(||f(h, r, t)||₁ − c)) ] + 𝔼_neg[ log(1 + exp(−||f(h, r, t)||₁)) ]
         where:
           - f(h, r, t) is the embedding-based scoring function (typically vector difference)
           - ||·||₁ is the L1 norm (sum of absolute values)
           - c is the confidence score for each positive triple
        """
        
        pos_scores = torch.norm(self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2]), p=1, dim=1)
        neg_scores = torch.norm(self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2]), p=1, dim=1)
        
        
        loss_pos = torch.mean(confidence_scores * F.softplus(pos_scores))
        loss_neg = torch.mean(F.softplus(-neg_scores))
        return loss_pos + loss_neg
    
    def gaussian_nll_loss(self, pos_triples, confidence_scores):
        
        """
        Calculates the Gaussian negative log-likelihood loss:
        L = 𝔼_pos[ (||f(h, r, t)||₁ − c)² / (2σ²) + log(σ²) ]
        where:
        - f(h, r, t) is the model's score for the triple (typically a distance in embedding space)
        - c is the confidence score (treated as the target mean)
        - σ² = 1 / c (inverse-confidence acts as variance estimate)
        """
        
        pos_scores = torch.norm(self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2]), p=1, dim=1)
        sigma_sq = 1 / (confidence_scores + 1e-8)  # Prevent division by zero
        loss = torch.mean((pos_scores - confidence_scores) ** 2 / (2 * sigma_sq) + torch.log(sigma_sq))
        return loss
    
    def contrastive_loss(self, pos_triples, neg_triples, margin=1.0):
        
        """
        Calculates the margin-based contrastive loss:
        L = 𝔼[ max(0, ||f(h⁺, r, t⁺)||₁ − ||f(h⁻, r, t⁻)||₁ + γ) ]
        where:
        - f(h, r, t) is the scoring function (typically distance in embedding space)
        - γ is the margin hyperparameter (default 1.0)
        - h⁺, r, t⁺ are from positive triples; h⁻, r, t⁻ are from negative triples
        """
        
        pos_scores = torch.norm(self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2]), p=1, dim=1)
        neg_scores = torch.norm(self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2]), p=1, dim=1)
        loss = torch.mean(F.relu(pos_scores - neg_scores + margin))
        return loss
    
    def kl_divergence_loss(self, pos_triples, confidence_scores):
        """
        Calculates the Kullback–Leibler (KL) divergence between predicted and target probability distributions:
        L = D_KL(P || Q) = Σ_i P(i) · log(P(i) / Q(i))
        where:
        - P = softmax(−confidence_scores) is the target distribution (based on confidence)
        - Q = softmax(−||f(h, r, t)||₁) is the model's predicted distribution over scores
        - f(h, r, t) is the scoring function (lower = better)
        """

        
        pos_scores = torch.norm(self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2]), p=1, dim=1)
        pos_probs = F.softmax(-pos_scores, dim=0)
        target_probs = F.softmax(-confidence_scores, dim=0)
        loss = F.kl_div(pos_probs.log(), target_probs, reduction='batchmean')
        return loss
   
class DistMultUncertainty(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(DistMultUncertainty, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
    
    def forward(self, h, r, t):
        head_embedding = self.entity_embeddings(h)
        relation_embedding = self.relation_embeddings(r)
        tail_embedding = self.entity_embeddings(t)
        return torch.sum(head_embedding * relation_embedding * tail_embedding, dim=1)

    def loss(self, pos_triples, neg_triples, confidence_scores, margin=1.0):
        pos_score = self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
        neg_score = self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2])
        pos_loss = confidence_scores * torch.clamp(margin - pos_score + neg_score, min=0)
        return pos_loss.sum()
    
    def loss_neg(self, pos_triples, neg_triples, pos_confidence_scores, neg_confidence_scores, margin=1.0):
        # Get scores (higher = better), so we negate them to make them similar to distance (lower = better)
        pos_scores = -self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
        neg_scores = -self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2])

        num_neg_samples = len(neg_scores) // len(pos_scores)
        pos_scores = pos_scores.repeat_interleave(num_neg_samples)
        pos_confidence_scores = pos_confidence_scores.repeat_interleave(num_neg_samples)

        pos_loss = torch.sum(pos_confidence_scores * torch.clamp(margin + pos_scores - neg_scores, min=0))
        neg_loss = torch.sum(neg_confidence_scores * torch.clamp(margin + pos_scores - neg_scores, min=0))

        return pos_loss + neg_loss

    
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

        pos_loss = -confidence_scores * F.logsigmoid(pos_scores)
        neg_loss = -F.logsigmoid(-neg_scores)

        return pos_loss + neg_loss

    def gaussian_nll_loss(self, pos_triples, confidence_scores):
        pos_scores = self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
        
        variance = confidence_scores + 1e-6  # Prevent division by zero
        
        return torch.mean(0.5 * torch.log(variance) + 0.5 * ((pos_scores - confidence_scores) ** 2) / variance)

    def contrastive_loss(self, pos_triples, neg_triples, margin=1.0):
        pos_scores = self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
        neg_scores = self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2])

        return torch.mean(F.relu(margin - pos_scores + neg_scores))

    def kl_divergence_loss(self, pos_triples, confidence_scores):
        pos_scores = self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])

        p = F.softmax(pos_scores, dim=0)
        q = F.softmax(confidence_scores, dim=0)

        return F.kl_div(p.log(), q, reduction='batchmean')
    
class ComplExUncertainty(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(ComplExUncertainty, self).__init__()
        self.entity_re_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.entity_im_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_re_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.relation_im_embeddings = nn.Embedding(num_relations, embedding_dim)

    def forward(self, h, r, t):
        head_real, head_imag = self.entity_re_embeddings(h), self.entity_im_embeddings(h)
        relation_real, relation_imag = self.relation_re_embeddings(r), self.relation_im_embeddings(r)
        tail_real, tail_imag = self.entity_re_embeddings(t), self.entity_im_embeddings(t)

        return torch.sum(
            head_real * relation_real * tail_real + head_imag * relation_real * tail_imag + head_real * relation_imag * tail_imag - head_imag * relation_imag * tail_real,
            dim=1
        )

    def loss(self, pos_triples, neg_triples, confidence_scores, margin=1.0):
        pos_score = self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
        neg_score = self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2])

        loss = confidence_scores * torch.clamp(margin - pos_score + neg_score, min=0)
        return loss.mean() 
    
    def loss_neg(self, pos_triples, neg_triples, pos_confidence_scores, neg_confidence_scores, margin=1.0):
        pos_scores = -self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
        neg_scores = -self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2])

        num_neg_samples = len(neg_scores) // len(pos_scores)
        pos_scores = pos_scores.repeat_interleave(num_neg_samples)
        pos_confidence_scores = pos_confidence_scores.repeat_interleave(num_neg_samples)

        pos_loss = pos_confidence_scores * torch.clamp(margin + pos_scores - neg_scores, min=0)
        neg_loss = neg_confidence_scores * torch.clamp(margin + pos_scores - neg_scores, min=0)

        return pos_loss.mean() + neg_loss.mean()
    
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

        pos_loss = -confidence_scores * F.logsigmoid(pos_score + 1e-8)
        neg_loss = -F.logsigmoid(-neg_score - 1e-8)

        return (pos_loss + neg_loss).mean()
    
    def gaussian_nll_loss(self, pos_triples, confidence_scores):
        pos_scores = self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])

        variance = confidence_scores  # Assume confidence scores represent variance
        loss = (pos_scores - confidence_scores) ** 2 / (2 * variance) + torch.log(variance + 1e-8)

        return loss.mean()
    
    def contrastive_loss(self, pos_triples, neg_triples, margin=1.0):
        pos_score = self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
        neg_score = self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2])

        loss = F.relu(margin - pos_score + neg_score)
        return loss.mean()
    
    def kl_divergence_loss(self, pos_triples, confidence_scores):
        pos_scores = self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
        
        p = F.softmax(pos_scores + 1e-8, dim=0)  # Model’s predicted distribution
        q = F.softmax(confidence_scores, dim=0)  # Confidence score as target

        return F.kl_div(p.log(), q, reduction="batchmean")

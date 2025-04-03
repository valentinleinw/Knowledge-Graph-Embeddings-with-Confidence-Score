import torch.nn as nn
import torch
import torch.nn.functional as F


class TransEUncertainty(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TransEUncertainty, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
    
    def forward(self, h, r, t):
        return self.entity_embeddings(h) + self.relation_embeddings(r) - self.entity_embeddings(t)
    
    def loss(self, pos_triples, neg_triples, confidence_scores, margin=1.0):
        pos_loss = torch.sum(confidence_scores * torch.clamp(
            margin + torch.norm(self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2]), p=1, dim=1) -
            torch.norm(self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2]), p=1, dim=1), min=0))
        return pos_loss
    
    def loss_neg(self, pos_triples, neg_triples, pos_confidence_scores, neg_confidence_scores, margin=1.0):

        # Compute positive and negative scores
        pos_scores = torch.norm(self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2]), p=1, dim=1)
        neg_scores = torch.norm(self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2]), p=1, dim=1)

        # Compute loss with confidence weighting
        pos_loss = torch.sum(pos_confidence_scores * torch.clamp(margin + pos_scores - neg_scores, min=0))
        neg_loss = torch.sum(neg_confidence_scores * torch.clamp(margin + pos_scores - neg_scores, min=0)) 

        total_loss = pos_loss + neg_loss
        return total_loss
    
    def objective_function(self, pos_triples, neg_triples, confidence_scores):
        """
        Implements the objective function:
            J = Sum( (f(l) - s_l)^2 ) for positive triples
              + Sum( psi_gamma(f(l))^2 ) for negative triples
        """

        # Compute the scores for positive and negative triples
        pos_scores = torch.norm(self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2]), p=1, dim=1)
        neg_scores = torch.norm(self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2]), p=1, dim=1)

        # First term: MSE loss for positive triples (f(l) - s_l)^2
        loss_pos = torch.mean(confidence_scores * (pos_scores - confidence_scores) ** 2)

        # Second term: Apply transformation psi_gamma for negative triples
        loss_neg = torch.mean(F.relu(neg_scores) ** 2)  # psi_gamma(f(l))^2

        # Total objective function
        return loss_pos + loss_neg
    
    def softplus_loss(self, pos_triples, neg_triples, confidence_scores):
        pos_scores = torch.norm(self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2]), p=1, dim=1)
        neg_scores = torch.norm(self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2]), p=1, dim=1)
        loss_pos = torch.mean(torch.log(1 + torch.exp(pos_scores - confidence_scores)))
        loss_neg = torch.mean(torch.log(1 + torch.exp(-neg_scores)))
        return loss_pos + loss_neg
    
    def gaussian_nll_loss(self, pos_triples, confidence_scores):
        pos_scores = torch.norm(self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2]), p=1, dim=1)
        sigma_sq = 1 / (confidence_scores + 1e-8)  # Prevent division by zero
        loss = torch.mean((pos_scores - confidence_scores) ** 2 / (2 * sigma_sq) + torch.log(sigma_sq))
        return loss
    
    def contrastive_loss(self, pos_triples, neg_triples, margin=1.0):
        pos_scores = torch.norm(self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2]), p=1, dim=1)
        neg_scores = torch.norm(self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2]), p=1, dim=1)
        loss = torch.mean(F.relu(pos_scores - neg_scores + margin))
        return loss
    
    def kl_divergence_loss(self, pos_triples, confidence_scores):
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
    
    def objective_function(self, pos_triples, neg_triples, confidence_scores):
        pos_scores = self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
        neg_scores = self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2])

        # Apply the objective function
        loss_pos = torch.mean(confidence_scores * (pos_scores - confidence_scores) ** 2)
        loss_neg = torch.mean(F.relu(neg_scores) ** 2)  # psi_gamma(f(l))^2

        return loss_pos + loss_neg
    
    def softplus_loss(self, pos_triples, neg_triples, confidence_scores):
        pos_scores = self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
        neg_scores = self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2])

        pos_loss = -torch.mean(confidence_scores * F.softplus(pos_scores))
        neg_loss = torch.mean(F.softplus(neg_scores))

        return pos_loss + neg_loss

    def gaussian_nll_loss(self, pos_triples, confidence_scores):
        pos_scores = self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
        
        mean = pos_scores
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
    
    def objective_function(self, pos_triples, neg_triples, confidence_scores):
        pos_scores = self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
        neg_scores = self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2])

        loss_pos = torch.mean(confidence_scores * (pos_scores - confidence_scores) ** 2)
        loss_neg = torch.mean(F.relu(neg_scores) ** 2)

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

class RotatEUncertainty(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(RotatEUncertainty, self).__init__()
        assert embedding_dim % 2 == 0, "Embedding dimension must be even for complex embeddings"
        self.embedding_dim = embedding_dim // 2  # Since we're using complex embeddings

        # Entity embeddings (real and imaginary parts)
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        nn.init.uniform_(self.entity_embeddings.weight, -0.1, 0.1)

        # Relation embeddings (rotation angle in radians)
        self.relation_embeddings = nn.Embedding(num_relations, self.embedding_dim)
        nn.init.uniform_(self.relation_embeddings.weight, -3.14, 3.14)  # Initialize between -pi and pi

    def forward(self, h, r, t):
        # Get embeddings
        head = self.entity_embeddings(h)  # (batch, embedding_dim)
        tail = self.entity_embeddings(t)  # (batch, embedding_dim)
        relation = self.relation_embeddings(r)  # (batch, embedding_dim / 2)

        # Split into real and imaginary parts
        head_real, head_imag = torch.chunk(head, 2, dim=1)
        tail_real, tail_imag = torch.chunk(tail, 2, dim=1)

        # Compute rotation (cosine and sine)
        cos_r = torch.cos(relation)
        sin_r = torch.sin(relation)

        # Rotate head entity
        rotated_head_real = head_real * cos_r - head_imag * sin_r
        rotated_head_imag = head_real * sin_r + head_imag * cos_r

        # Compute L2 distance between rotated head and tail
        score = (rotated_head_real - tail_real) ** 2 + (rotated_head_imag - tail_imag) ** 2
        score = torch.sum(score, dim=1)

        return score

    def loss(self, pos_triples, neg_triples, confidence_scores, margin=1.0):
        # Compute scores
        pos_score = self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
        neg_score = self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2])

        # Compute margin-based loss with uncertainty
        loss = F.relu(confidence_scores * (pos_score - neg_score + margin))

        return loss.mean()
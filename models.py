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
    
class RotatEUncertainty(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(RotatEUncertainty, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

    def forward(self, h, r, t):
        # Get the real embeddings of head, relation, and tail
        head_real = self.entity_embeddings(h)
        tail_real = self.entity_embeddings(t)
        relation_real = self.relation_embeddings(r)

        # Rotate the head embedding by adding the relation embedding
        rotated_head_real = head_real + relation_real  # This is the "rotation"

        # Compute the score as the L2 distance between the rotated head and tail
        score = torch.sum((rotated_head_real - tail_real)**2, dim=1)
        return score

    def loss(self, pos_triples, neg_triples, confidence_scores, margin=1.0):
        # Calculate the scores for positive and negative triples
        pos_score = self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
        neg_score = self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2])

        # Compute the margin-based loss with uncertainty (confidence scores)
        loss = torch.max(confidence_scores * (pos_score - neg_score + margin), torch.tensor(0.0, device=pos_score.device))

        # Return the mean loss
        return loss.mean()
    
    def objective_function(self, pos_triples, neg_triples, confidence_scores):
        pos_scores = self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
        neg_scores = self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2])

        loss_pos = torch.mean(confidence_scores * (pos_scores - confidence_scores) ** 2)
        loss_neg = torch.mean(F.relu(neg_scores) ** 2)

        return loss_pos + loss_neg
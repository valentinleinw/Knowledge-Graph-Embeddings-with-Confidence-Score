import torch.nn as nn
import torch
import math


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
    
class RotatEUncertainty(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(RotatEUncertainty, self).__init__()
        self.entity_re_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.entity_im_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_re_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.relation_im_embeddings = nn.Embedding(num_relations, embedding_dim)

    def forward(self, h, r, t):
        # Get the real and imaginary embeddings of head, relation, and tail
        head_real, head_imag = self.entity_re_embeddings(h), self.entity_im_embeddings(h)
        tail_real, tail_imag = self.entity_re_embeddings(t), self.entity_im_embeddings(t)
        relation_real, relation_imag = self.relation_re_embeddings(r), self.relation_im_embeddings(r)

        # Perform rotation in the complex plane
        rotated_tail_real = tail_real * torch.cos(relation_real) - tail_imag * torch.sin(relation_real)
        rotated_tail_imag = tail_real * torch.sin(relation_real) + tail_imag * torch.cos(relation_imag)

        # Compute the score as the distance between head + relation and tail in complex space
        score = torch.sum((head_real - rotated_tail_real)**2 + (head_imag - rotated_tail_imag)**2, dim=1)
        return score

    def loss(self, pos_triples, neg_triples, confidence_scores, margin=1.0):
        # Calculate the scores for positive and negative triples
        pos_score = self(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
        neg_score = self(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2])

        # Compute the loss with uncertainty (confidence scores)
        loss = confidence_scores * torch.clamp(margin + pos_score - neg_score, min=0)
        return loss.mean()
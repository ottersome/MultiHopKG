from typing import Any, Union
import numpy as np
import os
import sys
from pathlib import Path
import pytest

from torch.utils.data import DataLoader
import torch

from multihopkg.utils.data_splitting import TripleIds
from multihopkg.datasets import TestDataset
from tests.conftest import ValidatorDict

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from multihopkg.knowledge_graph import KGEModel

def test_embedding_dimensions_match_kg(
    knowledge_graph, entityRFD2id: dict[str, int], relationRFD2id: dict[str, int]
):
    """Test that embedding dimensions match knowledge graph entity/relation counts."""
    # Check entity count
    assert knowledge_graph.entity_embedding.shape[0] == len(entityRFD2id.keys()), \
        f"Entity embedding count ({knowledge_graph.entity_embeddings.shape[0]}) does not match the entity2id dictionary length ({len(entityRFD2id.keys())})"
    
    # Check relation count
    assert knowledge_graph.relation_embedding.shape[0] == len(relationRFD2id.keys()), \
        f"Relation embedding count ({knowledge_graph.relation_embeddings.shape[0]}) does not match the relation2id dictionary length ({len(relationRFD2id.keys())})"

def test_performance(
    test_triples: TripleIds,
    all_true_triples: TripleIds,
    knowledge_graph: KGEModel,
    model_config: dict,
    validation_thresholds: ValidatorDict[Union[int, float]],
):
    # Simulate namespace/class args
    class TempArgs:
        def __init__(self, nentity, nrelation, model_config, validation_thresholds):
            self.nentity = nentity
            self.nrelation = nrelation
            self.model_model = model_config
            self.validation_thresholds = validation_thresholds
            self.countries = False
            self.test_batch_size = model_config["test_batch_size"]
            self.test_log_steps = 100

            # Hardcoding these for no
            self.cpu_num = 10
            self.cuda = True

    args = TempArgs(knowledge_graph.nentity, knowledge_graph.nrelation, model_config, validation_thresholds)

    knowledge_graph.test_step(knowledge_graph, test_triples, all_true_triples, args)
    

# def test_link_prediction_hits_at_10(knowledge_graph: KGEModel, performance_threshold, test_triples: TripleIds):
#     """Test link prediction performance using Hits@10 metric."""
#         # Sample a subset of test examples for quick evaluation
#     test_samples = min(100, len(knowledge_graph.test_set))
#     test_triples = knowledge_graph.test_set[:test_samples]
#     
#     # Track hits@10
#     hits_at_10 = 0
#     
#     # Test link prediction for tail entity
#     for head, relation, tail in test_triples:
#         # Get indices
#         head_idx = knowledge_graph.entity_to_idx[head]
#         relation_idx = knowledge_graph.relation_to_idx[relation]
#         tail_idx = knowledge_graph.entity_to_idx[tail]
#         
#         # Get embeddings
#         head_emb = knowledge_graph.entity_embeddings[head_idx]
#         relation_emb = knowledge_graph.relation_embeddings[relation_idx]
#         
#         # Predict tail entities
#         scores = knowledge_graph.score_all_tails(head_emb, relation_emb)
#         
#         # Get top 10 predictions
#         top_indices = np.argsort(scores)[-10:]
#         
#         # Check if true tail is in top 10
#         if tail_idx in top_indices:
#             hits_at_10 += 1
#     
#     # Calculate hits@10 ratio
#     hits_at_10_ratio = hits_at_10 / test_samples
#     
#     # Check against threshold
#     assert hits_at_10_ratio >= performance_threshold, \
#         f"Hits@10 performance ({hits_at_10_ratio:.4f}) is below threshold ({performance_threshold})"
        
# TODO: Consider if something similar to this is necessary/helpful
# def test_embedding_similarity_distribution(embedding_model):
#     """Test that embedding similarity distribution looks reasonable."""
#     # Sample a subset of entities for quick evaluation
#     sample_size = min(1000, embedding_model.entity_embeddings.shape[0])
#     entity_indices = np.random.choice(embedding_model.entity_embeddings.shape[0], sample_size, replace=False)
#     
#     # Compute pairwise cosine similarities
#     sampled_embeddings = embedding_model.entity_embeddings[entity_indices]
#     norms = np.linalg.norm(sampled_embeddings, axis=1, keepdims=True)
#     normalized_embeddings = sampled_embeddings / norms
#     similarities = np.dot(normalized_embeddings, normalized_embeddings.T)
#     
#     # Get off-diagonal elements (pairwise similarities)
#     mask = np.ones_like(similarities, dtype=bool)
#     np.fill_diagonal(mask, 0)
#     similarities = similarities[mask]
#     
#     # Check that similarities are in reasonable range
#     assert -1.0 <= np.min(similarities) <= 1.0, "Minimum similarity is outside valid range"
#     assert -1.0 <= np.max(similarities) <= 1.0, "Maximum similarity is outside valid range"
#     
#     # Check that embeddings aren't all identical or all orthogonal
#     mean_sim = np.mean(similarities)
#     assert 0.0 <= mean_sim <= 0.9, f"Mean similarity ({mean_sim}) suggests embeddings may be too similar"
#     assert -0.1 <= mean_sim, f"Mean similarity ({mean_sim}) suggests embeddings may be too dissimilar"
#     
#     # Check standard deviation isn't too small
#     std_sim = np.std(similarities)
#     assert std_sim >= 0.05, f"Similarity standard deviation ({std_sim}) is too small, embeddings may lack diversity"

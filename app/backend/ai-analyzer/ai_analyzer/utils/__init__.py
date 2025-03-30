"""Utility functions and classes for the AI Analyzer."""

from ai_analyzer.utils.singleton_resources import get_qdrant_client, get_embedding_model, ResourceManager
from ai_analyzer.utils.resilience import search_qdrant_safely, CircuitBreaker
from ai_analyzer.utils.vector_db import update_tenant_vector_db
from ai_analyzer.utils.singleton_embeddings import SingletonEmbeddings

__all__ = [
    'get_qdrant_client',
    'get_embedding_model',
    'ResourceManager',
    'search_qdrant_safely',
    'CircuitBreaker',
    'update_tenant_vector_db',
    'SingletonEmbeddings'
]

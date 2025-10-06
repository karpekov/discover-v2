import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from typing import List, Dict, Any, Optional
from pathlib import Path


class CachedTextEncoder(nn.Module):
    """
    Text encoder that uses precomputed embeddings for fast training.
    Falls back to computing embeddings on-the-fly for unseen captions.
    """
    
    def __init__(self, cache_path: str, fallback_encoder: Optional[nn.Module] = None):
        super().__init__()
        
        self.cache_path = cache_path
        self.fallback_encoder = fallback_encoder
        
        # Load the cache
        print(f"Loading text embeddings cache from: {cache_path}")
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        self.caption_to_embedding = cache_data['caption_to_embedding']
        self.cache_config = cache_data.get('config', {})
        self.text_encoder_config = cache_data.get('text_encoder_config', {})
        self.num_cached_captions = cache_data.get('num_captions', len(self.caption_to_embedding))
        self.embedding_dim_base = cache_data.get('embedding_dim_base', 768)
        self.embedding_dim_projected = cache_data.get('embedding_dim_projected', 512)
        
        print(f"Loaded cache with {self.num_cached_captions} precomputed embeddings")
        print(f"  Base embedding dim: {self.embedding_dim_base}")
        print(f"  Projected embedding dim: {self.embedding_dim_projected}")
        
        # Stats tracking
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Create a learnable projection head if needed (for continued training)
        use_proj_head = self.text_encoder_config.get('use_text_proj_head', True)
        if use_proj_head:
            self.proj_head = nn.Linear(self.embedding_dim_base, self.embedding_dim_projected, bias=False)
            # Initialize with near-identity if possible
            with torch.no_grad():
                if self.embedding_dim_projected <= self.embedding_dim_base:
                    identity_init = torch.eye(self.embedding_dim_base)[:self.embedding_dim_projected]
                    self.proj_head.weight.copy_(identity_init)
                else:
                    nn.init.xavier_uniform_(self.proj_head.weight)
        else:
            self.proj_head = None
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass compatible with existing TextEncoder API.
        Note: This method is not the primary interface for cached embeddings.
        """
        raise NotImplementedError(
            "CachedTextEncoder uses precomputed embeddings. "
            "Use encode_texts() or encode_texts_clip() instead of forward()."
        )
    
    def forward_clip(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        CLIP-style forward pass compatible with existing API.
        Note: This method is not the primary interface for cached embeddings.
        """
        raise NotImplementedError(
            "CachedTextEncoder uses precomputed embeddings. "
            "Use encode_texts_clip() instead of forward_clip()."
        )
    
    def encode_texts(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """
        Encode texts using cached embeddings (base embeddings).
        
        Args:
            texts: List of text strings
            device: Target device
            
        Returns:
            embeddings: [len(texts), embedding_dim_base] L2-normalized
        """
        if not texts:
            return torch.empty(0, self.embedding_dim_base, device=device)
        
        embeddings = []
        
        for text in texts:
            text = text.strip()
            if text in self.caption_to_embedding:
                # Cache hit - use precomputed embedding
                cached_data = self.caption_to_embedding[text]
                embedding = cached_data['base'].to(device)
                embeddings.append(embedding)
                self.cache_hits += 1
            else:
                # Cache miss - compute on the fly or use fallback
                if self.fallback_encoder is not None:
                    embedding = self.fallback_encoder.encode_texts([text], device)[0]
                    embeddings.append(embedding)
                else:
                    # No fallback - use zero embedding (should rarely happen in practice)
                    embedding = torch.zeros(self.embedding_dim_base, device=device)
                    embeddings.append(embedding)
                    print(f"Warning: Cache miss for text: '{text[:50]}...' (using zero embedding)")
                
                self.cache_misses += 1
        
        return torch.stack(embeddings)
    
    def encode_texts_clip(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """
        Encode texts using cached embeddings with CLIP projection.
        
        Args:
            texts: List of text strings
            device: Target device
            
        Returns:
            embeddings: [len(texts), embedding_dim_projected] L2-normalized
        """
        if not texts:
            return torch.empty(0, self.embedding_dim_projected, device=device)
        
        embeddings = []
        
        for text in texts:
            text = text.strip()
            if text in self.caption_to_embedding:
                # Cache hit - use precomputed embedding
                cached_data = self.caption_to_embedding[text]
                
                if self.proj_head is not None:
                    # Use base embedding and apply learnable projection
                    base_embedding = cached_data['base'].to(device)
                    projected = self.proj_head(base_embedding)
                    projected = F.normalize(projected, p=2, dim=-1)
                    embeddings.append(projected)
                else:
                    # Use precomputed projected embedding
                    embedding = cached_data['projected'].to(device)
                    embeddings.append(embedding)
                
                self.cache_hits += 1
            else:
                # Cache miss - compute on the fly or use fallback
                if self.fallback_encoder is not None:
                    embedding = self.fallback_encoder.encode_texts_clip([text], device)[0]
                    embeddings.append(embedding)
                else:
                    # No fallback - use zero embedding
                    embedding = torch.zeros(self.embedding_dim_projected, device=device)
                    embeddings.append(embedding)
                    print(f"Warning: Cache miss for text: '{text[:50]}...' (using zero embedding)")
                
                self.cache_misses += 1
        
        return torch.stack(embeddings)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache hit/miss statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'num_cached_captions': self.num_cached_captions
        }
    
    def print_cache_stats(self):
        """Print cache statistics."""
        stats = self.get_cache_stats()
        print(f"Text Embedding Cache Statistics:")
        print(f"  Cache hits: {stats['cache_hits']}")
        print(f"  Cache misses: {stats['cache_misses']}")
        print(f"  Hit rate: {stats['hit_rate']:.2%}")
        print(f"  Cached captions: {stats['num_cached_captions']}")
    
    def get_output_dim(self) -> int:
        """Get the output embedding dimension."""
        return self.embedding_dim_projected if self.proj_head is not None else self.embedding_dim_base
    
    def __repr__(self):
        return (
            f"CachedTextEncoder(\n"
            f"  cache_path={self.cache_path},\n"
            f"  cached_captions={self.num_cached_captions},\n"
            f"  base_dim={self.embedding_dim_base},\n"
            f"  projected_dim={self.embedding_dim_projected},\n"
            f"  has_proj_head={self.proj_head is not None},\n"
            f"  has_fallback={self.fallback_encoder is not None}\n"
            f")"
        )

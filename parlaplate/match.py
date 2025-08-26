"""
Embedding utilities and candidate matching for menu items.
"""
import os
import hashlib
from typing import List, Dict, Any
import json

import numpy as np
from openai import OpenAI

from .schemas import MenuItem, MenuJSON
from .utils import price_bucket


def build_item_text(item: MenuItem) -> str:
    """
    Build searchable text representation of a menu item.
    
    Args:
        item: MenuItem to process
        
    Returns:
        Combined text string for embedding
    """
    parts = [item.name]
    
    if item.ingredients:
        parts.extend(item.ingredients)
    
    if item.keywords:
        parts.extend(item.keywords)
    
    if item.category:
        parts.append(item.category)
    
    return " ".join(parts).lower()


def embed_texts(client: OpenAI, model: str, texts: List[str]) -> np.ndarray:
    """
    Get embeddings for a list of texts.
    
    Args:
        client: OpenAI client
        model: Embedding model name
        texts: List of texts to embed
        
    Returns:
        Numpy array of embeddings (texts x embedding_dim)
    """
    response = client.embeddings.create(
        input=texts,
        model=model
    )
    
    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings)


def get_embedding_cache_path(menu_json: MenuJSON, base_dir: str = "opt/menu_content") -> str:
    """
    Get cache file path for menu embeddings based on content hash.
    
    Args:
        menu_json: Menu data
        base_dir: Base directory for cache files
        
    Returns:
        Path to embedding cache file
    """
    # Create hash of menu content
    content_str = json.dumps(
        [item.model_dump() for item in menu_json.items],
        sort_keys=True
    )
    content_hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]
    
    restaurant_name = menu_json.restaurant.name or "unknown"
    safe_name = "".join(c for c in restaurant_name if c.isalnum() or c in "-_").lower()
    
    return os.path.join(base_dir, f"{safe_name}_{content_hash}.emb.npy")


def load_or_compute_embeddings(
    client: OpenAI,
    model_embed: str,
    menu_json: MenuJSON,
    force_recompute: bool = False
) -> np.ndarray:
    """
    Load embeddings from cache or compute them if needed.
    
    Args:
        client: OpenAI client
        model_embed: Embedding model name
        menu_json: Menu data
        force_recompute: Force recomputation even if cache exists
        
    Returns:
        Item embeddings array
    """
    cache_path = get_embedding_cache_path(menu_json)
    
    # Try to load from cache
    if not force_recompute and os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception:
            pass  # Fall back to recomputing
    
    # Compute embeddings
    texts = [build_item_text(item) for item in menu_json.items]
    embeddings = embed_texts(client, model_embed, texts)
    
    # Cache the results
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, embeddings)
    except Exception:
        pass  # Caching failed, but embeddings are computed
    
    return embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between query and item embeddings.
    
    Args:
        a: Query embedding (1D array)
        b: Item embeddings (2D array: items x embedding_dim)
        
    Returns:
        Similarity scores for each item
    """
    # Normalize vectors
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    
    # Compute cosine similarity
    return np.dot(b_norm, a_norm)


def apply_constraints_filter(
    items: List[MenuItem],
    constraints: Dict[str, Any]
) -> List[bool]:
    """
    Apply constraint filters to menu items.
    
    Args:
        items: List of menu items
        constraints: Constraint dictionary
        
    Returns:
        Boolean mask indicating which items pass filters
    """
    mask = [True] * len(items)
    
    avoid_allergens = constraints.get("avoid_allergens", [])
    diet_requirements = constraints.get("diet", [])
    
    for i, item in enumerate(items):
        # Filter out items with avoided allergens
        if avoid_allergens:
            item_allergens = [a.lower() for a in item.allergens]
            if any(allergen.lower() in item_allergens for allergen in avoid_allergens):
                mask[i] = False
                continue
        
        # Apply dietary filters (basic implementation)
        if diet_requirements:
            item_keywords = [k.lower() for k in item.keywords]
            item_ingredients = [k.lower() for k in item.ingredients]
            item_text = " ".join(item_keywords + item_ingredients + [item.name.lower()])
            
            for diet in diet_requirements:
                diet = diet.lower()
                if diet == "vegetarian":
                    # Simple heuristic: avoid obvious meat items
                    meat_terms = ["chicken", "beef", "lamb", "fish", "seafood", "turkey", "pork"]
                    if any(term in item_text for term in meat_terms):
                        mask[i] = False
                        break
                elif diet == "vegan":
                    # Simple heuristic: avoid meat and dairy
                    non_vegan = ["chicken", "beef", "lamb", "fish", "seafood", "cheese", "milk", "cream", "butter"]
                    if any(term in item_text for term in non_vegan):
                        mask[i] = False
                        break
    
    return mask


def rank_candidates(
    client: OpenAI,
    model_embed: str,
    menu_json: MenuJSON,
    query_keywords: List[str],
    constraints: Dict[str, Any],
    top_k: int = 8
) -> List[MenuItem]:
    """
    Rank menu items based on query and constraints.
    
    Args:
        client: OpenAI client
        model_embed: Embedding model name
        menu_json: Menu data
        query_keywords: Keywords from user query
        constraints: Filtering constraints
        top_k: Number of top items to return
        
    Returns:
        Ranked list of menu items
    """
    if not menu_json.items:
        return []
    
    # Load or compute item embeddings
    item_embeddings = load_or_compute_embeddings(client, model_embed, menu_json)
    
    # Build query text from keywords and constraints
    query_parts = query_keywords.copy()
    
    # Add dietary preferences to query
    if constraints.get("diet"):
        query_parts.extend(constraints["diet"])
    
    # Add price preference hint
    price_pref = constraints.get("price_preference")
    if price_pref:
        query_parts.append(f"price-{price_pref}")
    
    query_text = " ".join(query_parts)
    
    # Get query embedding
    if query_text.strip():
        query_embedding = embed_texts(client, model_embed, [query_text])[0]
    else:
        # If no query, use random/default ranking
        similarities = np.random.rand(len(menu_json.items))
    
    if query_text.strip():
        # Compute similarities
        similarities = cosine_similarity(query_embedding, item_embeddings)
    
    # Apply constraint filters
    constraint_mask = apply_constraints_filter(menu_json.items, constraints)
    
    # Apply price preference (de-prioritize expensive items for "low" preference)
    if price_pref == "low":
        for i, item in enumerate(menu_json.items):
            if item.price and price_bucket(item.price) == "high":
                similarities[i] *= 0.7  # Reduce score for expensive items
    
    # Combine similarity scores with constraint mask
    final_scores = np.where(constraint_mask, similarities, -1.0)
    
    # Get top-k indices
    top_indices = np.argsort(final_scores)[::-1][:top_k]
    
    # Filter out items that didn't pass constraints
    valid_indices = [i for i in top_indices if final_scores[i] >= 0]
    
    return [menu_json.items[i] for i in valid_indices]
"""
Helper functions for AI Engineering course.
Reusable utilities for search, embeddings, and evaluation.

Usage:
    from helpers import (
        load_wands_products, load_wands_queries, load_wands_labels,
        snowball_tokenize, build_index, score_bm25,
        calculate_ndcg, evaluate_search,
        get_embedding_openai, get_embedding_local, batch_embed_local,
        cosine_similarity, batch_cosine_similarity
    )
"""

import string
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# DATA LOADING
# ============================================================


def load_wands_products(data_dir: str = "../data") -> pd.DataFrame:
    """
    Load WANDS products from local file.

    Args:
        data_dir: Path to the data directory containing wayfair-products.csv

    Returns:
        DataFrame with product information including product_id, product_name,
        product_class, category_hierarchy, product_description, etc.
    """
    filepath = Path(data_dir) / "wayfair-products.csv"
    products = pd.read_csv(filepath, sep="\t")
    products = products.rename(columns={"category hierarchy": "category_hierarchy"})
    return products


def load_wands_queries(data_dir: str = "../data") -> pd.DataFrame:
    """
    Load WANDS queries from local file.

    Args:
        data_dir: Path to the data directory containing wayfair-queries.csv

    Returns:
        DataFrame with query_id and query columns
    """
    filepath = Path(data_dir) / "wayfair-queries.csv"
    return pd.read_csv(filepath, sep="\t")


def load_wands_labels(data_dir: str = "../data") -> pd.DataFrame:
    """
    Load WANDS relevance labels from local file.

    Args:
        data_dir: Path to the data directory containing wayfair-labels.csv

    Returns:
        DataFrame with query_id, product_id, label (Exact/Partial/Irrelevant),
        and grade (2/1/0) columns
    """
    filepath = Path(data_dir) / "wayfair-labels.csv"
    labels = pd.read_csv(filepath, sep="\t")
    grade_map = {"Exact": 2, "Partial": 1, "Irrelevant": 0}
    labels["grade"] = labels["label"].map(grade_map)
    return labels


# ============================================================
# TOKENIZATION
# ============================================================

# Lazy load stemmer to avoid import time overhead
_stemmer = None
_punct_trans = str.maketrans({key: " " for key in string.punctuation})


def _get_stemmer():
    """Lazy load the Snowball stemmer."""
    global _stemmer
    if _stemmer is None:
        import Stemmer

        _stemmer = Stemmer.Stemmer("english")
    return _stemmer


def snowball_tokenize(text: str) -> list[str]:
    """
    Tokenize text with Snowball stemming.

    Converts text to lowercase, removes punctuation, splits on whitespace,
    and applies Snowball stemming to each token.

    Args:
        text: The text to tokenize

    Returns:
        List of stemmed tokens
    """
    if pd.isna(text) or text is None:
        return []
    text = str(text).translate(_punct_trans)
    tokens = text.lower().split()
    stemmer = _get_stemmer()
    return [stemmer.stemWord(token) for token in tokens]


# ============================================================
# BM25 SEARCH
# ============================================================


def build_index(docs: list[str], tokenizer=None) -> tuple[dict, list[int]]:
    """
    Build an inverted index from a list of documents.

    Args:
        docs: List of document strings to index
        tokenizer: Function that takes text and returns list of tokens
                   (defaults to snowball_tokenize)

    Returns:
        index: dict mapping term -> {doc_id: term_count}
        doc_lengths: list of document lengths (in tokens)
    """
    if tokenizer is None:
        tokenizer = snowball_tokenize

    index = {}
    doc_lengths = []

    for doc_id, doc in enumerate(docs):
        tokens = tokenizer(doc)
        doc_lengths.append(len(tokens))
        term_counts = Counter(tokens)

        for term, count in term_counts.items():
            if term not in index:
                index[term] = {}
            index[term][doc_id] = count

    return index, doc_lengths


def get_tf(term: str, doc_id: int, index: dict) -> int:
    """
    Get term frequency for a term in a document.

    Args:
        term: The term to look up
        doc_id: The document ID
        index: The inverted index

    Returns:
        Term frequency (count), or 0 if not found
    """
    return index.get(term, {}).get(doc_id, 0)


def get_df(term: str, index: dict) -> int:
    """
    Get document frequency for a term.

    Args:
        term: The term to look up
        index: The inverted index

    Returns:
        Number of documents containing the term
    """
    return len(index.get(term, {}))


def bm25_idf(df: int, num_docs: int) -> float:
    """
    BM25 IDF formula.

    Args:
        df: Document frequency
        num_docs: Total number of documents

    Returns:
        IDF score
    """
    return np.log((num_docs - df + 0.5) / (df + 0.5) + 1)


def bm25_tf(
    tf: int, doc_len: int, avg_doc_len: float, k1: float = 1.2, b: float = 0.75
) -> float:
    """
    BM25 TF normalization.

    Args:
        tf: Term frequency
        doc_len: Document length in tokens
        avg_doc_len: Average document length
        k1: Saturation parameter (default 1.2)
        b: Length normalization (default 0.75)

    Returns:
        Normalized TF score
    """
    return (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))


def score_bm25(
    query: str,
    index: dict,
    num_docs: int,
    doc_lengths: list[int],
    tokenizer=None,
    k1: float = 1.2,
    b: float = 0.75,
) -> np.ndarray:
    """
    Score all documents using BM25.

    Args:
        query: The search query
        index: Inverted index
        num_docs: Total number of documents
        doc_lengths: List of document lengths
        tokenizer: Tokenization function (defaults to snowball_tokenize)
        k1: BM25 saturation parameter
        b: BM25 length normalization parameter

    Returns:
        Array of scores for each document
    """
    if tokenizer is None:
        tokenizer = snowball_tokenize

    query_tokens = tokenizer(query)
    scores = np.zeros(num_docs)
    avg_doc_len = np.mean(doc_lengths) if doc_lengths else 1.0

    for token in query_tokens:
        df = get_df(token, index)
        if df == 0:
            continue

        idf = bm25_idf(df, num_docs)

        if token in index:
            for doc_id, tf in index[token].items():
                tf_norm = bm25_tf(tf, doc_lengths[doc_id], avg_doc_len, k1, b)
                scores[doc_id] += idf * tf_norm

    return scores


def search_bm25(
    query: str,
    index: dict,
    products_df: pd.DataFrame,
    doc_lengths: list[int],
    tokenizer=None,
    k: int = 10,
) -> pd.DataFrame:
    """
    Search products using BM25 and return top-k results.

    Args:
        query: The search query
        index: Inverted index
        products_df: DataFrame of products
        doc_lengths: Document lengths
        tokenizer: Tokenization function
        k: Number of results to return

    Returns:
        DataFrame with top-k products and scores
    """
    scores = score_bm25(query, index, len(products_df), doc_lengths, tokenizer)
    top_k_idx = np.argsort(-scores)[:k]

    results = products_df.iloc[top_k_idx].copy()
    results["bm25_score"] = scores[top_k_idx]
    results["rank"] = range(1, k + 1)
    return results


# ============================================================
# EVALUATION
# ============================================================


def calculate_dcg(relevances: list[int], k: int = 10) -> float:
    """
    Calculate Discounted Cumulative Gain at position k.

    Args:
        relevances: List of relevance grades (0, 1, 2)
        k: Number of positions to consider

    Returns:
        DCG score
    """
    relevances = relevances[:k]
    dcg = 0.0
    for i, rel in enumerate(relevances):
        dcg += (2**rel - 1) / np.log2(i + 2)
    return dcg


def calculate_ndcg(relevances: list[int], k: int = 10) -> float:
    """
    Calculate Normalized DCG at position k.

    Args:
        relevances: List of relevance grades (0, 1, 2)
        k: Number of positions to consider

    Returns:
        NDCG score (0 to 1)
    """
    dcg = calculate_dcg(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0.0


def get_relevance_grades(
    product_ids: list[int], query_id: int, labels_df: pd.DataFrame
) -> list[int]:
    """
    Get relevance grades for a list of product IDs given a query.

    Args:
        product_ids: List of product IDs in rank order
        query_id: The query ID
        labels_df: DataFrame with relevance labels

    Returns:
        List of relevance grades (0, 1, or 2) for each product
    """
    query_labels = labels_df[labels_df["query_id"] == query_id]
    label_dict = dict(zip(query_labels["product_id"], query_labels["grade"]))
    return [label_dict.get(pid, 0) for pid in product_ids]


def evaluate_search(
    search_func,
    products_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    k: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Evaluate search across all queries.

    Args:
        search_func: Function that takes query string and returns DataFrame with product_id
        products_df: DataFrame of products
        queries_df: DataFrame of queries
        labels_df: DataFrame with relevance labels
        k: Number of results to consider
        verbose: Whether to print progress

    Returns:
        DataFrame with query_id, query, and ndcg columns
    """
    results = []

    for _, row in queries_df.iterrows():
        query_id = row["query_id"]
        query_text = row["query"]

        search_results = search_func(query_text)
        product_ids = search_results["product_id"].tolist()[:k]
        relevances = get_relevance_grades(product_ids, query_id, labels_df)
        ndcg = calculate_ndcg(relevances, k)

        results.append({"query_id": query_id, "query": query_text, "ndcg": ndcg})

    results_df = pd.DataFrame(results)

    if verbose:
        print(f"Evaluated {len(results_df)} queries")
        print(f"Mean NDCG@{k}: {results_df['ndcg'].mean():.4f}")

    return results_df


# ============================================================
# EMBEDDINGS - API-BASED (OpenAI via LiteLLM)
# ============================================================


def get_embedding_openai(
    text: str, model: str = "text-embedding-3-small"
) -> np.ndarray:
    """
    Get embedding using OpenAI API via LiteLLM.

    Args:
        text: Text to embed
        model: OpenAI embedding model name

    Returns:
        Embedding vector as numpy array
    """
    import litellm

    response = litellm.embedding(model=model, input=[text])
    return np.array(response.data[0]["embedding"])


def batch_embed_openai(
    texts: list[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
    verbose: bool = True,
) -> np.ndarray:
    """
    Embed multiple texts using OpenAI API with batching.

    Args:
        texts: List of texts to embed
        model: OpenAI embedding model name
        batch_size: Number of texts per API call
        verbose: Whether to print progress

    Returns:
        Embeddings as numpy array of shape (len(texts), embedding_dim)
    """
    import litellm

    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = litellm.embedding(model=model, input=batch)
        batch_embs = [d["embedding"] for d in response.data]
        embeddings.extend(batch_embs)

        if verbose and (i + batch_size) % 500 == 0:
            print(f"Embedded {min(i + batch_size, len(texts))}/{len(texts)} texts")

    return np.array(embeddings)


# ============================================================
# EMBEDDINGS - LOCAL (Hugging Face / Sentence Transformers)
# ============================================================

# Cache the model to avoid reloading
_local_model_cache = {}


def get_local_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Load a local embedding model from Hugging Face (cached).

    Args:
        model_name: Hugging Face model name

    Returns:
        SentenceTransformer model instance
    """
    if model_name not in _local_model_cache:
        from sentence_transformers import SentenceTransformer

        print(f"Loading model '{model_name}' from Hugging Face...")
        _local_model_cache[model_name] = SentenceTransformer(model_name)
        dim = _local_model_cache[model_name].get_sentence_embedding_dimension()
        print(f"Model loaded! Embedding dimension: {dim}")
    return _local_model_cache[model_name]


def get_embedding_local(text: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Get embedding using a local Hugging Face model.

    Args:
        text: Text to embed
        model_name: Hugging Face model name

    Returns:
        Embedding vector as numpy array
    """
    model = get_local_model(model_name)
    return model.encode(text, convert_to_numpy=True)


def batch_embed_local(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Embed multiple texts using a local Hugging Face model.

    Args:
        texts: List of texts to embed
        model_name: Hugging Face model name
        batch_size: Number of texts per batch
        show_progress: Whether to show progress bar

    Returns:
        Embeddings as numpy array of shape (len(texts), embedding_dim)
    """
    model = get_local_model(model_name)
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )


# ============================================================
# SIMILARITY & SEARCH
# ============================================================


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity (between -1 and 1)
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def batch_cosine_similarity(query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
    """
    Calculate cosine similarity between query and all documents.

    Args:
        query_emb: Query embedding vector
        doc_embs: Document embeddings matrix (num_docs, embedding_dim)

    Returns:
        Array of similarity scores for each document
    """
    query_norm = query_emb / np.linalg.norm(query_emb)
    doc_norms = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)
    return doc_norms @ query_norm


def semantic_search(
    query: str,
    product_embeddings: np.ndarray,
    products_df: pd.DataFrame,
    embed_func=None,
    k: int = 10,
) -> pd.DataFrame:
    """
    Search products using embedding similarity.

    Args:
        query: The search query
        product_embeddings: Pre-computed product embeddings
        products_df: DataFrame of products
        embed_func: Function to embed the query (defaults to get_embedding_local)
        k: Number of results to return

    Returns:
        DataFrame with top-k products and similarity scores
    """
    if embed_func is None:
        embed_func = get_embedding_local

    query_emb = embed_func(query)
    similarities = batch_cosine_similarity(query_emb, product_embeddings)

    top_k_idx = np.argsort(-similarities)[:k]
    results = products_df.iloc[top_k_idx].copy()
    results["similarity"] = similarities[top_k_idx]
    results["rank"] = range(1, k + 1)
    return results


# ============================================================
# UTILITY
# ============================================================


def get_product_sample(
    products_df: pd.DataFrame, n: int = 5000, seed: int = 42
) -> pd.DataFrame:
    """
    Get a consistent sample of products.

    Using the same seed ensures all students work with the same sample.

    Args:
        products_df: Full products DataFrame
        n: Number of products to sample
        seed: Random seed for reproducibility

    Returns:
        Sampled DataFrame with reset index
    """
    return products_df.sample(n=n, random_state=seed).reset_index(drop=True)


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """
    Normalize scores to [0, 1] range using min-max normalization.

    Args:
        scores: Array of scores

    Returns:
        Normalized scores
    """
    min_score = scores.min()
    max_score = scores.max()
    if max_score - min_score == 0:
        return np.zeros_like(scores)
    return (scores - min_score) / (max_score - min_score)

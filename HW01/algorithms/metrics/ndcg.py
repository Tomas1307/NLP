from .dcg import dcg_at_k

def ndcg_at_k(relevance_query: list, k: int):
    dcg = dcg_at_k(relevance_query, k)
    # Ordenamos las relevancias en orden descendente para obtener el IDCG
    sorted_relevance_query = sorted(relevance_query, reverse=True)
    idcg = dcg_at_k(sorted_relevance_query, k)
    ndcg = dcg / idcg if idcg > 0 else 0
    return ndcg

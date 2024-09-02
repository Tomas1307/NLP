def ndcg_at_k(relevance_query: list,k: int):
    dcg = dcg_at_k(relevance_query, k)
    relevance_query.sort(reverse=True)
    idcg = dcg_at_k(relevance_query, k)
    ndcg = dcg/idcg
    return ndcg

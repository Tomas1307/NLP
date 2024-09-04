from .dcg import dcg_at_k

def ndcg_at_k(relevance_query: list, k: int):
    """
    Función que calcula la métrica ndcg para un k determinado

    Args
    relevance_query: Query de entrada con relevancia de número natural
    k: Número k deseado de la consulta
    """
    dcg = dcg_at_k(relevance_query, k)
    sorted_relevance_query = sorted(relevance_query, reverse=True)
    idcg = dcg_at_k(sorted_relevance_query, k)
    ndcg = dcg / idcg if idcg > 0 else 0
    return ndcg

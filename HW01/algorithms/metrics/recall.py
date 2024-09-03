def recall_at_k(relevance_query: list , number_relevant_docs:int, k:int):
    query_at_k = relevance_query[:k]
    relevant = query_at_k.count(1)
    return(relevant/number_relevant_docs)

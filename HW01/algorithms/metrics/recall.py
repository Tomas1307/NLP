def recall_at_k(relevance_query: list , number_relevant_docs:int, k:int):
    """
      Función que calcula la métrica de recall para una relevance query en una pila k
      Args
      relevance_query: Documentos con relevancia binaria
      k: Número k deseado
      number_relevant_docs: Número de documentos relevantes
      """
    query_at_k = relevance_query[:k]
    relevant = query_at_k.count(1)
    return(relevant/number_relevant_docs)

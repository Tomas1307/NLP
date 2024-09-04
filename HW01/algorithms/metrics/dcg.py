import numpy as np

def dcg_at_k(relevance_query, k):
    """
      Función que calcula la métrica de dcg para un vector de documentos en una pila k

      Args
      relevance_query: Documentos con relevancia de número natural
      k: Número k deseado
      """
    relevance_query = np.array(relevance_query)[:k]
    discounts = np.log2(np.maximum(np.arange(1, k + 1), 2))
    dcg = np.sum(relevance_query / discounts)
    return dcg

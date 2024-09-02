import numpy as np


def precision(relevance_query: list): 
      list_of_documents = np.array(relevance_query)
      #Sacar numero de documentos irrelevantes
      number_of_zeros = np.sum(list_of_documents == 0)
      #Sacar número de documentos relevantes
      number_of_ones = np.sum(list_of_documents == 1)

      #Proporcion de documentos que son relevantes
      precision_calculated = number_of_ones/(number_of_zeros+number_of_ones)
      return precision_calculated
    
def precision_at_k(relevance_query: list ,k: int):
    #Resolver error en el cual k puede ser mayor que el tamaño de la lista
    list_of_documents = np.array(relevance_query)
    k = min(k,len(list_of_documents))

    #Solo revisa los k documentos
    k_documents = list_of_documents[:k]

    #Numero de documentos irrelevantes
    number_of_zeros = np.sum(k_documents == 0)

    #Numero de documentos relevantes
    number_of_ones = np.sum(k_documents == 1)

    #Proporcion de documentos que son relevantes (siguiendo misma logica de la precision normal)
    precision_calculated =  number_of_ones/(number_of_zeros+number_of_ones)

    return precision_calculated



class BinarySearch:
    def __init__(self) -> None:
        pass

    def intersect_two_lists(self, p1, p2):
        """
        Perform the merge operation (intersection) between two posting lists.

        Args:
            p1 (list): Posting list of term 1 (sorted list of document IDs).
            p2 (list): Posting list of term 2 (sorted list of document IDs).

        Returns:
            list: A list of document IDs that are present in both p1 and p2.
        """
        answer = []
        i, j = 0, 0

        while i < len(p1) and j < len(p2):
            if p1[i] == p2[j]:
                answer.append(p1[i])
                i += 1
                j += 1
            elif p1[i] < p2[j]:
                i += 1
            else:
                j += 1

        return answer

    
    def intersect_multiple_lists(self, lists):
        """
        Realiza la intersección de múltiples listas.
        
        Args:
            lists (list of lists): Lista de listas de posting a intersectar.
        
        Returns:
            list: Lista resultante de la intersección de todas las listas de entrada.
        """
        if not lists:
            return []
        
        if len(lists) == 1:
            return lists[0]
        
        result = lists[0]
        for i in range(1, len(lists)):
            result = self.intersect_two_lists(result, lists[i])
        
        return result
    
    def not_word(self,p,inv_index):
        """
        Realiza múltiples intersecciones alrededor de todo el índice

        Args:
        p: Palabra de la cual se desea obtener los documentos en los cuales no aparece
        index: Todo el índice invertido
        """
        #Obtiene todos los documentos de la palabra de entrada
        docs_with_word = inv_index.get(p, [])


        docs_without_word = []


        for i, docs in inv_index.items():
            if i != p:
                for doc in docs:
                    if doc not in docs_with_word:
                        #Se añade la palabra a la respuesta si todos documentos NO coinciden con los documentos de la palabra de entrada
                        if doc not in docs_without_word:
                            docs_without_word.append(doc)
        
        return {p: sorted(docs_without_word)}
    
    def generate_results_file(self, queries_df, inverted_index, output_file):
        """
        Genera un archivo de resultados con el formato especificado.
        
        Args:
            queries_df (DataFrame): DataFrame con las consultas.
            inverted_index (dict): Índice invertido.
            output_file (str): Nombre del archivo de salida.
        """
        with open(output_file, 'w') as f:
            for i, row in queries_df.iterrows():
                query_id = f"q{i+1:02d}"  # Asumiendo que los query_id empiezan en q01
                query_terms = row['query_list']
                
                index_values = [inverted_index[term] for term in query_terms if term in inverted_index]
                
                if index_values:
                    sorted_lists = [sorted(sublist) for sublist in index_values]
                    result = self.intersect_multiple_lists(sorted_lists)
                else:
                    result = []
                
                if result:
                    doc_list = ','.join([f"d{int(doc):03d}" for doc in result])
                    f.write(f"{query_id} {doc_list}\n")
                else:
                    f.write(f"{query_id}\n")
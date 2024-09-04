import pandas as pd
import json
from gensim import corpora
from gensim.models import TfidfModel
from gensim.matutils import cossim
from algorithms.binary_search.inverted_index import InvertedIndex
from algorithms.binary_search.query_processor import QueryProcessor
from algorithms.binary_search.text_processor import TextProcessor
import logging
import time
import numpy as np

class RRDVGensim:
    """
    A class for processing an inverted index and creating a TF-IDF matrix using Gensim.
    """

    def __init__(self):
        """
        Initialize the RRDVGensim class.
        """
        self.inverted_index = InvertedIndex()
        self.inverted_index_occurrences = None
        self.dictionary = None
        self.tfidf_model = None
        self.logger = self.setup_logger()

    def setup_logger(self):
        """
        Set up the logger for the class.
        """
        logger = logging.getLogger('RRDVGensim')
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        return logger

    def load_inverted_index(self, file_path: str ="inverted_index.json"):
        """
        Load the inverted index from a JSON file or create a new one if the file doesn't exist.

        Parameters:
        file_path (str): The path to the JSON file containing the inverted index. Default is "inverted_index.json".

        Returns:
        None
        """
        self.logger.info("Loading inverted index...")
        try:
            with open(file_path) as file:
                self.inverted_index_occurrences = json.load(file)
            self.logger.info("Inverted index loaded successfully.")
        except FileNotFoundError:
            self.logger.warning(f"File not found: {file_path}")
            self.logger.info("Creating new inverted index...")
            self.inverted_index_occurrences = self.inverted_index.inverted_index_complete_pipeline(occurrences=True,filename=file_path)
            self.logger.info("New inverted index created successfully.")
            

    def process_inverted_index(self):
        """
        Process the inverted index into a pandas DataFrame.

        Returns:
        pandas.DataFrame: A DataFrame containing the processed inverted index with columns 'termino', 'df', and 'Postings'.
        """
        data = []
        for key, postings in self.inverted_index_occurrences.items():
            termino, df = key.strip('()').split(',')
            df = int(df)
            data.append((termino, df, postings))
        return pd.DataFrame(data, columns=['termino', 'df', 'Postings'])

    def create_documents(self, df: pd.DataFrame):
        """
        Create a list of documents from the processed inverted index DataFrame.

        Parameters:
        df (pandas.DataFrame): The processed inverted index DataFrame.

        Returns:
        list: A list of documents, where each document is a list of terms.
        """
        documentos = {}
        for index, row in df.iterrows():
            termino = row['termino']
            df_value = row['df']
            for doc_id in row['Postings']:
                if doc_id not in documentos:
                    documentos[doc_id] = []
                documentos[doc_id].extend([termino] * df_value)
        return [terminos for doc_id, terminos in sorted(documentos.items())]

    def create_tfidf_matrix(self, inverted_index: dict):
        """
        Create a TF-IDF matrix from the given inverted index.

        Parameters:
        inverted_index (dict): The inverted index dictionary.

        Returns:
        tuple: A tuple containing:
            - corpus_tfidf (list of gensim.interfaces.TransformedCorpus): The TF-IDF matrix.
            - dictionary (gensim.corpora.Dictionary): The dictionary mapping words to their ids.
            - df (pandas.DataFrame): The processed inverted index DataFrame with an additional 'term_id' column.
        """
        self.inverted_index_occurrences = inverted_index
        df = self.process_inverted_index()
        documentos_list = self.create_documents(df)

        self.dictionary = corpora.Dictionary(documentos_list)
        df['term_id'] = df['termino'].map(lambda x: self.dictionary.token2id.get(x, -1))
        corpus = [self.dictionary.doc2bow(doc) for doc in documentos_list]
        self.tfidf_model = TfidfModel(corpus)
        corpus_tfidf = self.tfidf_model[corpus]

        return corpus_tfidf

    def cosine_similarity(self, doc1: list, doc2: list):
        """
        Calculate the cosine similarity between two document vectors.

        Parameters:
        doc1 (list of str): The first document as a list of words.
        doc2 (list of str): The second document as a list of words.

        Returns:
        float: The cosine similarity between the two documents.
        """
        if self.tfidf_model is None or self.dictionary is None:
            raise ValueError("TF-IDF model and dictionary must be created before calculating cosine similarity. Call create_tfidf_matrix first.")

        # Convert documents to bag-of-words representation
        vec1 = self.dictionary.doc2bow(doc1)
        vec2 = self.dictionary.doc2bow(doc2)

        # Convert to TF-IDF representation
        vec1_tfidf = self.tfidf_model[vec1]
        vec2_tfidf = self.tfidf_model[vec2]

        # Calculate cosine similarity
        similarity = cossim(vec1_tfidf, vec2_tfidf)

        return similarity
    
    def sorted_query(self, query_list):
        resultado = []
        
        for item in query_list:
            query, documentos_str = item.split(' ', 1)
            documentos = documentos_str.split(',')
            
            documentos_ordenados = sorted(documentos, key=lambda x: float(x.split(':')[1]), reverse=True)
            
            resultado.append(f"{query} {','.join(documentos_ordenados)}")
        
        return resultado
    
    
    def format_cosine_similarities(self, df_queries, df_texts):
        """
        Format cosine similarities between queries and texts.

        Parameters:
        df_queries (pd.DataFrame): DataFrame containing processed queries.
        df_texts (pd.DataFrame): DataFrame containing processed texts.

        Returns:
        str: Formatted string of cosine similarities.
        """
        self.logger.info("Calculating cosine similarities...")
        results = {}
        total_comparisons = len(df_queries) * len(df_texts)
        comparisons_done = 0

        for i in range(len(df_queries)):
            query_id = df_queries['identifier'].iloc[i]
            query_results = []

            for j in range(len(df_texts)):
                value = self.cosine_similarity(df_queries["query_list"].iloc[i], df_texts["text_list"].iloc[j])
                if value > 0:
                    doc_id = f"d{df_texts['identifier'].iloc[j]}"
                    query_results.append(f"{doc_id}:{value:.6f}")
                
                comparisons_done += 1
                if comparisons_done % 100 == 0:  # Log progress every 100 comparisons
                    progress = (comparisons_done / total_comparisons) * 100
                    self.logger.info(f"Progress: {progress:.2f}% ({comparisons_done}/{total_comparisons})")

            results[query_id] = ','.join(query_results)

        formatted_output = []
        for query_id, result in results.items():
            formatted_output.append(f"{query_id} {result}")
            
        formatted_output = self.sorted_query(formatted_output)

        self.logger.info("Cosine similarities calculation completed.")
        return '\n'.join(formatted_output)

    def process_and_save_results(self, output_filename: str = "results/GENSIM-consultas_resultado.txt"):
        """
        Process queries and texts, calculate cosine similarities, and save results to a file.

        Parameters:
        queries_file (str): Path to the queries file.
        texts_directory (str): Path to the directory containing text files.
        output_filename (str): Name of the output file to save results.

        Returns:
        None
        """
        start_time = time.time()
        self.logger.info("Starting RRDV process...")

        # Load or create inverted index
        self.load_inverted_index()
        self.logger.info("25% complete - Inverted index loaded/created")

        # Create TF-IDF matrix
        self.logger.info("Creating TF-IDF matrix...")
        self.create_tfidf_matrix(self.inverted_index_occurrences)
        self.logger.info("50% complete - TF-IDF matrix created")

        # Process queries and texts
        self.logger.info("Processing queries and texts...")
        query_processor = QueryProcessor()
        text_processor = TextProcessor()

        df_queries = query_processor.process_queries()
        df_texts = text_processor.process_texts()
        self.logger.info("75% complete - Queries and texts processed")

        # Calculate and format cosine similarities
        self.logger.info("Calculating cosine similarities...")
        formatted_result = self.format_cosine_similarities(df_queries, df_texts)

        # Save results to file
        self.logger.info(f"Saving results to {output_filename}...")
        with open(output_filename, 'w') as f:
            f.write(formatted_result)

        end_time = time.time()
        total_time = end_time - start_time
        self.logger.info(f"100% complete - Results saved in {output_filename}")
        self.logger.info(f"Total processing time: {total_time:.2f} seconds")
        
        
    def count_documents(self, value):
        """
        Counts the number of documents in a comma-separated string of document identifiers.

        Args:
            value (str): A string containing document identifiers separated by commas.

        Returns:
            int: The number of documents in the string.
        """
        documents = value.split(',')
        return len(documents)

    def load_relevance_judgments(self, filepath):
        """
        Loads relevance judgments from a TSV file and creates a relevance dictionary.

        Args:
            filepath (str): The path to the relevance judgments TSV file.

        Returns:
            tuple: A tuple containing:
                - df (pandas.DataFrame): A DataFrame with the queries and their associated documents.
                - relevance_dict (dict): A dictionary where each query ID maps to a dictionary of documents and their relevance scores.
        """
        df = pd.read_csv(filepath, sep='\t', header=None, names=['query', 'value'])
        df['M'] = df['value'].apply(self.count_documents)
        
        relevance_dict = {}
        for index, row in df.iterrows():
            query_id = row['query']
            relevances = {doc.split(':')[0]: int(doc.split(':')[1]) for doc in row['value'].split(',')}
            relevance_dict[query_id] = relevances
            
        return df, relevance_dict

    def precision_at_m(self, retrieved_docs, relevant_docs, M):
        """
        Calculates Precision at M (P@M).

        Args:
            retrieved_docs (list): A list of retrieved document identifiers.
            relevant_docs (dict): A dictionary of relevant document identifiers for a query.
            M (int): The number of top documents to consider for the metric.

        Returns:
            float: The precision at M.
        """
        relevant_retrieved = [doc for doc in retrieved_docs[:M] if doc in relevant_docs]
        return len(relevant_retrieved) / M

    def recall_at_m(self, retrieved_docs, relevant_docs, M):
        """
        Calculates Recall at M (R@M).

        Args:
            retrieved_docs (list): A list of retrieved document identifiers.
            relevant_docs (dict): A dictionary of relevant document identifiers for a query.
            M (int): The number of top documents to consider for the metric.

        Returns:
            float: The recall at M.
        """
        relevant_retrieved = [doc for doc in retrieved_docs[:M] if doc in relevant_docs]
        return len(relevant_retrieved) / len(relevant_docs)

    def dcg_at_m(self, retrieved_docs, relevance_scores, M):
        """
        Calculates Discounted Cumulative Gain at M (DCG@M).

        Args:
            retrieved_docs (list): A list of retrieved document identifiers.
            relevance_scores (dict): A dictionary of relevance scores for the retrieved documents.
            M (int): The number of top documents to consider for the metric.

        Returns:
            float: The DCG at M.
        """
        dcg = 0.0
        for i in range(M):
            rel_score = relevance_scores.get(retrieved_docs[i], 0)
            dcg += (2**rel_score - 1) / np.log2(i + 2)
        return dcg

    def ndcg_at_m(self, retrieved_docs, relevance_scores, M):
        """
        Calculates Normalized Discounted Cumulative Gain at M (NDCG@M).

        Args:
            retrieved_docs (list): A list of retrieved document identifiers.
            relevance_scores (dict): A dictionary of relevance scores for the retrieved documents.
            M (int): The number of top documents to consider for the metric.

        Returns:
            float: The NDCG at M.
        """
        dcg = self.dcg_at_m(retrieved_docs, relevance_scores, M)
        ideal_order = sorted(relevance_scores.values(), reverse=True)
        ideal_dcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_order[:M]))
        return dcg / ideal_dcg if ideal_dcg > 0 else 0

    def evaluate_queries(self, relevance_filepath: str = "data/relevance-judgments/relevance-judgments.tsv", gensim_results_filepath: str = "results/GENSIM-consultas_resultado.txt"):
        """
        Evaluates the performance of a retrieval system using relevance judgments and computes the P@M, R@M, and NDCG@M metrics for each query.

        Args:
            relevance_filepath (str): The path to the relevance judgments TSV file.
            gensim_results_filepath (str): The path to the Gensim results file.

        Returns:
            None
        """
        df, relevance_dict = self.load_relevance_judgments(relevance_filepath)
        
        gensim_results = {}
        with open(gensim_results_filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                query = parts[0]
                docs = [doc.split(':')[0] for doc in parts[1].split(',')]
                gensim_results[query] = docs
        
        queries = df['query'].tolist()
        for query in queries:
            retrieved_docs = gensim_results[query]
            relevant_docs = relevance_dict.get(query)
            M = len(relevant_docs)
            p_at_m = self.precision_at_m(retrieved_docs, relevant_docs, M)
            r_at_m = self.recall_at_m(retrieved_docs, relevant_docs, M)
            ndcg_at_m_score = self.ndcg_at_m(retrieved_docs, relevant_docs, M)
            
            print(f"Query: {query} -> P@M: {p_at_m}, R@M: {r_at_m}, NDCG@M: {ndcg_at_m_score}")


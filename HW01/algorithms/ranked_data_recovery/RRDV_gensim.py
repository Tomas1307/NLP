import pandas as pd
import json
from gensim import corpora
from gensim.models import TfidfModel
from gensim.matutils import cossim
from algorithms.binary_search.inverted_index import InvertedIndex

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

    def load_inverted_index(self, file_path: str ="inverted_index.json"):
        """
        Load the inverted index from a JSON file or create a new one if the file doesn't exist.

        Parameters:
        file_path (str): The path to the JSON file containing the inverted index. Default is "inverted_index.json".

        Returns:
        None
        """
        try:
            with open(file_path) as file:
                self.inverted_index_occurrences = json.load(file)
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            self.inverted_index_occurrences = self.inverted_index.inverted_index_complete_pipeline(occurrences=True)
            

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
import pandas as pd
import glob
from KafNafParserPy import KafNafParser
from utils_processor.processor import Processor
import logging
import time
from nltk import word_tokenize

class TextProcessor:
    def __init__(self):
        self.processor_ = Processor()
        self.logger = self.setup_logger()

    def setup_logger(self):
        logger = logging.getLogger('TextProcessor')
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        return logger

    def read_files(self, directory: str = "data/docs-raw-texts/") -> pd.DataFrame:
        self.logger.info(f"Reading files from directory: {directory}")
        start_time = time.time()

        files = glob.glob(directory + "wes2015.d*.naf")
        data = []

        for file in files:
            identifier = file.split(".")[-2][-3:]  
            naf_parser = KafNafParser(file)
            raw_text = naf_parser.get_raw()
            title = naf_parser.root.find('nafHeader/fileDesc').get('title')
            data.append({"identifier": identifier, "text": raw_text, "title": title})

        dataFrame = pd.DataFrame(data)

        end_time = time.time()
        self.logger.info(f"Finished reading files. Time taken: {end_time - start_time:.2f} seconds")

        return dataFrame

    def replace_title_on_text(self, dataFrame: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Replacing titles in text and cleaning newlines")
        start_time = time.time()

        for i in range(len(dataFrame["text"])):
            if dataFrame["title"][i] in dataFrame["text"][i]:
                dataFrame["text"][i] = dataFrame["text"][i].replace(dataFrame["title"][i] + ".", "")
            dataFrame["text"][i] = dataFrame["text"][i].replace("\n", "")

        end_time = time.time()
        self.logger.info(f"Finished replacing titles. Time taken: {end_time - start_time:.2f} seconds")

        return dataFrame

    def process_texts(self, directory: str = "data/docs-raw-texts/") -> pd.DataFrame:
        self.logger.info("Starting text processing pipeline")
        start_time = time.time()

        df = self.read_files(directory)
        df = self.replace_title_on_text(df)

        self.logger.info("Processing and normalizing text")
        df["text"] = df["text"] + " " + df["title"]
        df.drop(columns=["title"], inplace=True)

        for i in range(len(df["text"])):
            text = df["text"][i]
            processed_text = word_tokenize(text)
            processed_text = self.processor_.to_lowercase(processed_text)
            processed_text = self.processor_.remove_punctuation(processed_text)
            processed_text = self.processor_.remove_non_ascii(processed_text)
            processed_text = self.processor_.remove_stopwords(processed_text)
            processed_text = self.processor_.stem_verbs(processed_text)
            df.at[i, "text"] = ' '.join(processed_text)

        df["text_list"] = df["text"].apply(lambda x: x.split())

        end_time = time.time()
        self.logger.info(f"Completed text processing pipeline. Total time taken: {end_time - start_time:.2f} seconds")

        return df
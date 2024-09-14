import re
from nltk.tokenize import word_tokenize

class Processor:
    def __init__(self):
        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}(?:\.[A-Z|a-z]{2,})?\b'

    def remove_non_ascii(self, text):
        return ''.join(char for char in text if ord(char) < 128)

    def to_lowercase(self, text):
        return text.lower()

    def remove_escape_sequences(self, text):
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        text = ansi_escape.sub('', text)
        text = re.sub(r'\d+m', '', text)
        return text

    def replace_email(self, text):
        return re.sub(self.email_pattern, '<<EMAIL>>', text)

    def remove_brackets(self, text):
        return re.sub(r'\[|\]', '', text)

    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def split_sentences(self, text):
        # Split on period followed by space or end of string
        return re.split(r'(?<=\.)(?:\s+|\Z)', text)

    def process_sentence(self, sentence):
        #sentence = self.remove_non_ascii(sentence)
        sentence = self.to_lowercase(sentence)
        #sentence = self.remove_escape_sequences(sentence)
        sentence = self.replace_email(sentence)
        sentence = self.remove_brackets(sentence)
        sentence = self.clean_text(sentence)
        
        words = word_tokenize(sentence)
        words = [word for word in words if word not in ['', ':', '>', '<']]
        
        # Remove parentheses
        words = [word.replace("(", "").replace(")", "") for word in words]
        
        # Only create a sentence if there are words
        if words:
            return f"<s> {' '.join(words)} </s>"
        else:
            return ""  # Return an empty string if there are no words
    
    

    def process_text(self, text):
        sentences = self.split_sentences(text)
        processed_sentences = [self.process_sentence(sentence) for sentence in sentences]
        # Remove any empty strings that might have been returned
        processed_sentences = [s for s in processed_sentences if s]
        return " ".join(processed_sentences)
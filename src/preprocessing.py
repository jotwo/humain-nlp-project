import os
import re
import string
from io import StringIO

import pandas as pd
import spacy
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import CountVectorizer


class PDFCorpusPreprocessor:
    def __init__(self, reports_dir_path):
        self.reports_dir_path = reports_dir_path
        self.pdf_report_names = [
            r for r in os.listdir(self.reports_dir_path) if r.endswith('.pdf')
        ]
        self.corpus = None
        self.doc_term_matrix = None

    def extract_corpus_from_pdf(self):
        temp_corpus = {}
        with StringIO() as output_buffer:
            for name in self.pdf_report_names:
                content = extract_text(os.path.join(self.reports_dir_path, name))
                # key is the name minus 4 characters corresponding to '.pdf'
                key = name[:-4]
                temp_corpus[key] = [content]
        self.corpus = pd.DataFrame.from_dict(temp_corpus, orient='index')
        self.corpus.columns = ['content']

    def build_document_term_matrix(self, lemmatize=True):
        if self.corpus is not None:

            if lemmatize:
                corpus_to_clean = pd.DataFrame(
                    self.corpus['content'].apply(self.__lemmatize_text)
                )
            else:
                corpus_to_clean = self.corpus


            cleaned_corpus = pd.DataFrame(
                corpus_to_clean['content'].apply(self.__clean_report_content)
            )

            vectorizer = CountVectorizer(stop_words='english')
            count_vectorized_data = vectorizer.fit_transform(cleaned_corpus['content'])

            self.doc_term_matrix = pd.DataFrame(
                count_vectorized_data.toarray(),
                columns=vectorizer.get_feature_names(),
                index=self.corpus.index,
            )

        else:
            raise Exception(
                "You first have to extract a corpus to build a document-term matrix."
            )
    
    def get_document_term_matrix(self):
        if self.doc_term_matrix is not None:
            return self.doc_term_matrix.copy()

        else:
            raise Exception(
                "You first have to extract a corpus and build a document-term matrix before demanding to get one."
            )
    
    def get_corpus(self):
        if self.corpus is not None:
            return self.corpus.copy()

        else:
            raise Exception(
                "You first have to extract a corpus before demanding to get one."
            )

    def __clean_report_content(self, text):
        """Make text lowercase, remove text in square brackets, remove punctuation, remove words containing numbers and other unwanted substrings."""
        
        url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
        email_regex = "[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}"
        # remove URLs
        text = re.sub(url_regex, "", text)
        # remove emails
        text = re.sub(email_regex, "", text)
        # replace dashes by spaces
        text = re.sub("[-–_]", " ", text)
        # => to lower case letters
        text = text.lower()
        # remove punctuation
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        # remove numbers or "words" containing numbers
        text = re.sub("\w*\d\w*", "", text)
        # replace newline character with space
        text = re.sub("\n", " ", text)
        # remove greek characters
        text = re.sub("[α-ωΑ-Ω]*", "", text)
        # remove some remaining parts of math expressions
        text = re.sub("ˆ[A-Za-z]*", "", text)

        return text
    
    def __lemmatize_text(self, text):
        """Transform each token of the text to its lemma individually and returns it."""

        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        text = ' '.join([token.lemma_ for token in doc])

        return text

import os
import re
import string
from io import StringIO

import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import CountVectorizer


class PDFCorpusPreprocessor:
    def __init__(self, reports_dir_path='../data/reports'):
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        self.reports_dir_path = reports_dir_path
        self.pdf_report_names = [
            r for r in os.listdir(self.reports_dir_path) if r.endswith('.pdf')
        ]
        self.corpus = None
        self.doc_term_matrix = None
        self.i = 0



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

            cleaned_corpus = pd.DataFrame(
                self.corpus['content'].apply(self.__clean_report_content)
            )

            # if lemmatize:
            #     cleaned_corpus = cleaned_corpus['content'].apply(self.__lemmatize_text)

            vectorizer = CountVectorizer(stop_words='english')
            # vectorizer = CountVectorizer(stop_words='english', tokenizer=word_tokenize)
            count_vectorized_data = vectorizer.fit_transform(cleaned_corpus['content'])

            self.doc_term_matrix = pd.DataFrame(
                count_vectorized_data.toarray(),
                columns=vectorizer.get_feature_names(),
                index=self.corpus.index,
            )

            return self.doc_term_matrix

        else:
            raise Exception(
                "You first have to extract a corpus to build a document-term matrix"
            )
    
    def __lemmatize_text(self, text):
        """Transform each token of the text to its lemma individually and returns it."""
        lemmatizer = WordNetLemmatizer()
        if self.i == 0:
            print(text)
            print('\n')
        token_list = nltk.word_tokenize(text)
        if self.i == 0:
            print(token_list[2])
            print('\n')
        text = ' '.join([lemmatizer.lemmatize(t) for t in token_list])
        if self.i == 0:
            # print(text)
            # print('\n')
            print(lemmatizer.lemmatize(token_list[2]))
        self.i = self.i+1
        return text

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

        text = self.__lemmatize_text(text)

        return text

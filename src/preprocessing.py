import os
import re
import string
from io import StringIO

import pandas as pd
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import CountVectorizer


class PDFCorpusPreprocessor:
    def __init__(self, reports_dir_path='../data/reports'):
        self.reports_dir_path = reports_dir_path
        self.pdf_report_names = [r for r in os.listdir(self.reports_dir_path) if r.endswith('.pdf')]
        self.corpus = None
        
    
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

        self.corpus = pd.DataFrame(self.corpus['content'].apply(self.__clean_report_content))
    
    def build_document_term_matrix(self):
        if self.corpus is not None:
            vectorizer = CountVectorizer(stop_words='english')
            count_vectorized_data = vectorizer.fit_transform(self.corpus['content'])
            self.dtm = pd.DataFrame(count_vectorized_data.toarray(),
                                    columns=vectorizer.get_feature_names(),
                                    index=self.corpus.index)
            return self.dtm

        else:
            raise Exception('You first have to extract a corpus to build a document-term matrix')
    
    def __clean_report_content(self, text):
        '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
        url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
        email_regex = '[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}'
        # remove URLs
        text = re.sub(url_regex, '', text)
        # remove emails
        text = re.sub(email_regex, '', text)
        # replace dashes by spaces
        text = re.sub('[-–_]', ' ', text)
        # => to lower case letters
        text = text.lower()
        # remove punctuation
        text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
        # remove numbers or "words" containing numbers
        text = re.sub('\w*\d\w*', '', text)
        # replace newline chaacter with space
        text = re.sub('\n', ' ', text)
        return text


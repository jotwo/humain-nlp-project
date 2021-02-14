import os
import re
import string
from io import StringIO

import pandas as pd
import spacy
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import CountVectorizer


from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.layout import LAParams, LTTextBoxHorizontal
from pdfminer.converter import PDFPageAggregator
from pdfminer.pdfpage import PDFPage



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




class PDFCorpus:
    # def __init__(self, pdfs_dir_path):

    #     self.nlp = spacy.load('en_core_web_sm')

    #     self.pdfs_dir_path = pdfs_dir_path
        
    #     pdfs_names = [
    #         r for r in os.listdir(self.pdfs_dir_path) if r.endswith('.pdf')
    #     ]

    #     self.docs_df = pd.DataFrame(columns=['name'])
    #     self.docs_df.rename_axis('doc_id', axis='index', inplace=True)
    #     self.paragraphs_df = pd.DataFrame(columns=['doc_id', 'content'])
    #     self.paragraphs_df.rename_axis('paragraph_id', axis='index', inplace=True)
    #     self.sentences_df = pd.DataFrame(columns=['paragraph_id', 'content'])
    #     self.sentences_df.rename_axis('sentence_id', axis='index', inplace=True)
    #     self.words_df = pd.DataFrame(columns=['sentence_id', 'content'])
    #     self.words_df.rename_axis('word_id', axis='index', inplace=True)

    #     self._build_tables(pdfs_names)

    def __init__(self):

        self.nlp = spacy.load('en_core_web_sm')

        self.docs_df = pd.DataFrame(columns=['name'])
        self.docs_df.rename_axis('doc_id', axis='index', inplace=True)
        self.paragraphs_df = pd.DataFrame(columns=['doc_id', 'content'])
        self.paragraphs_df.rename_axis('paragraph_id', axis='index', inplace=True)
        self.sentences_df = pd.DataFrame(columns=['paragraph_id', 'content'])
        self.sentences_df.rename_axis('sentence_id', axis='index', inplace=True)
        self.words_df = pd.DataFrame(columns=['sentence_id', 'content'])
        self.words_df.rename_axis('word_id', axis='index', inplace=True)

        self.corpus_df = pd.DataFrame(columns=['doc_id', 'paragraph_id', 'sentence_id', 'token'])
        self.corpus_df.rename_axis('token_id', axis='index', inplace=True)

    def add_pdf(self, pdf_filepath):
        
        # at this stage we are supposed to deal with '.pdf' files only
        # => 4 last characters are corresponding to '.pdf' and we remove them
        pdf_name = os.path.basename(pdf_filepath)[:-4]

        self.docs_df =  self.docs_df.append({'name': pdf_name}, ignore_index=True)

        rsrcmgr = PDFResourceManager()

        device = PDFPageAggregator(rsrcmgr, laparams=LAParams())

        interpreter = PDFPageInterpreter(rsrcmgr, device)

        paragraphs_list = []

        with open(pdf_filepath, 'rb') as document:
            for page in PDFPage.get_pages(document):
                interpreter.process_page(page)
                layout = device.get_result()
                for element in layout:
                    if isinstance(element, LTTextBoxHorizontal):
                        paragraphs_list.append(element.get_text())
    
        new_doc_id = len(self.docs_df) -1
        self._add_to_tables(new_doc_id, paragraphs_list)


            
    def _add_to_tables(self, doc_id, paragraphs_list):
        
        # paragraphs_df = pd.DataFrame({'content': paragraphs_list})

        # print(paragraphs_df)
        # print(self.docs_df)

        if self.corpus_df.empty:
            paragraph_id = 0
            sentence_id = 0
            token_id = 0
        
        else:
            last_corpus_entry = self.corpus_df.iloc[-1]
            paragraph_id = last_corpus_entry['paragraph_id'] + 1
            sentence_id = last_corpus_entry['sentence_id'] + 1
            token_id = last_corpus_entry.name + 1
        
        for paragraph in paragraphs_list:
            paragraph_doc = self.nlp(paragraph)

            save = sentence_id 
            for sentence in paragraph_doc.sents:
                cleaned_sentence = self.__clean_content(sentence.text)
                if len(cleaned_sentence.strip()) != 0:
                    # print(cleaned_sent)
                    # print()
                    sentence_doc = self.nlp(cleaned_sentence)
                    for token in sentence_doc:
                        if len(token.text.strip()) != 0:
                            token_id += 1
                            print(f'doc_id: {doc_id},\tparagraph_id: {paragraph_id},\tsentence_id: {sentence_id},\t\ttoken_id: {token_id},\t\ttoken: {token.text}')
                    
                    sentence_id += 1

            if sentence_id != save:
                paragraph_id += 1

    

    def __clean_content(self, text):
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
        


    def _build_tables(self, pdfs_names):
        # at this stage we are sure to deal with '.pdf' files only
        # => 4 last characters are corresponding to '.pdf'
        names_whitout_extensions = [name[:-4] for name in pdfs_names]
        self.documents = pd.DataFrame(names_whitout_extensions, columns=['name'])

        self.paragraphs =  pd.DataFrame()

        doc_path = os.path.join(self.pdfs_dir_path, self.documents.loc[4, 'name'] + '.pdf')

        # paragraphs_list = self._extract_pdf(doc_path)

        self._extract_pdf(doc_path)


    def _extract_pdf(self, pdf_file_path):

        rsrcmgr = PDFResourceManager()

        device = PDFPageAggregator(rsrcmgr, laparams=LAParams())

        interpreter = PDFPageInterpreter(rsrcmgr, device)

        paragraphs_list = []

        with open(pdf_file_path, 'rb') as document:
            for page in PDFPage.get_pages(document):
                interpreter.process_page(page)
                layout = device.get_result()
                for element in layout:
                    if isinstance(element, LTTextBoxHorizontal):
                        paragraphs_list.append(element.get_text())


        paragraphs_df = pd.DataFrame({'content': paragraphs_list})
        print(paragraphs_df)

        sentences_df = pd.concat([self._get_sentences_from_paragraphs(id, row) for id, row in paragraphs_df.iterrows()], ignore_index=True)
        sentences_df.rename_axis('sentence_id', axis='index', inplace=True)
        print(sentences_df)

        tokens_df = pd.concat([self._get_tokens_from_sentences(id, row) for id, row in sentences_df.iterrows()], ignore_index=True)
        tokens_df.rename_axis('token_id', axis='index', inplace=True)
        print(tokens_df)

#         test = '''In the short run, however, even this robust growth in supply is likely to leave some 
# companies scrambling. It would be insufficient to meet the 12 percent annual growth in 
# demand that could result in the most aggressive case that we modeled (Exhibit 4). This 
# scenario would produce a shortfall of roughly 250,000 data scientists. As a result, we expect 
# to see salaries for data scientists continue to grow. However, one trend could mitigate 
# demand in the medium term: the possibility that some part of the activities performed 
# by data scientists may become automated. More than 50 percent of the average data 
# scientist’s work is data preparation, including cleaning and structuring data. As data tools 
# improve, they could perform a significant portion of these activities, potentially helping to 
# ease the demand for data scientists within ten years.
# '''
#         d = {'content': test}
#         self._get_sentences_from_paragraphs(4444, d)
        

    def _get_sentences_from_paragraphs(self, id, row):
        paragraph_doc = self.nlp(row['content'])
        sentences_list = [sent.text for sent in paragraph_doc.sents]
        paragraph_ids = [id] * len(sentences_list)
        sentences_df = pd.DataFrame( {'paragraph_id': paragraph_ids, 'content': sentences_list})
        sentences_df.rename_axis('sentence_id', axis='index', inplace=True)
        return sentences_df
    
    def _get_tokens_from_sentences(self, id, row):
        sentence_doc = self.nlp(row['content'])
        tokens_list = [token.text for token in sentence_doc]
        sentence_ids = [id] * len(tokens_list)
        tokens_df = pd.DataFrame( {'sentence_id': sentence_ids, 'token': tokens_list})
        tokens_df.rename_axis('token_id', axis='index', inplace=True)
        return tokens_df
        


















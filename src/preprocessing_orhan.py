import os
import re
import datetime as dt

import pandas as pd
import spacy

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.layout import LAParams, LTTextBoxHorizontal
from pdfminer.converter import PDFPageAggregator
from pdfminer.pdfpage import PDFPage


class PDFCorpus:
    def __init__(self):

        self.nlp = spacy.load("en_core_web_sm")

        self.docs_df = pd.DataFrame(columns=["date", "name"])
        self.docs_df.rename_axis("doc_id", axis="index", inplace=True)
        self.paragraphs_df = pd.DataFrame(columns=["doc_id", "paragraph"])
        self.paragraphs_df.rename_axis("paragraph_id", axis="index", inplace=True)

    def get_paragraphs_df(self):
        return self.paragraphs_df.copy()

    def get_docs_df(self):
        return self.docs_df.copy()

    def add_multiple_pdfs(self, pdfs_dir_path):
        # we only extract pdf files from this directory
        pdf_names = [r for r in os.listdir(pdfs_dir_path) if r.endswith(".pdf")]
        for pdf_name in pdf_names:
            self.add_pdf(os.path.join(pdfs_dir_path, pdf_name))

    def add_pdf(self, pdf_filepath):

        # at this stage we are supposed to deal with '.pdf' files only
        # => 4 last characters are corresponding to '.pdf' and we remove them
        pdf_name = os.path.basename(pdf_filepath)[:-4]

        today_date = pd.to_datetime(dt.date.today())

        self.docs_df = self.docs_df.append(
            {"date": today_date, "name": pdf_name}, ignore_index=True
        )
        self.docs_df.rename_axis("doc_id", axis="index", inplace=True)

        rsrcmgr = PDFResourceManager()

        device = PDFPageAggregator(rsrcmgr, laparams=LAParams())

        interpreter = PDFPageInterpreter(rsrcmgr, device)

        paragraphs_list = []

        with open(pdf_filepath, "rb") as document:
            for page in PDFPage.get_pages(document):
                interpreter.process_page(page)
                layout = device.get_result()
                for element in layout:
                    if isinstance(element, LTTextBoxHorizontal):
                        paragraphs_list.append(element.get_text())

        new_doc_id = len(self.docs_df) - 1

        self._add_to_tables(new_doc_id, paragraphs_list)

    def _add_to_tables(self, doc_id, paragraphs_list):

        paragraph_id = len(self.paragraphs_df)

        paragraphs_dict = {}

        for paragraph in paragraphs_list:
            paragraph_doc = self.nlp(paragraph)
            num_sentences = 1
            # for sentence in paragraph_doc.sents:
                # cleaned_sentence = self.__clean_content(sentence.text)
                # test if the cleaned sentence is not composed of whitespaces only
                # in that case we discard this meaningless sentence
                # if len(cleaned_sentence.strip()) != 0:
                #     num_sentences += 1
            # check if the paragraph wasn't composed only of discarded sentences
            # in that case, we also discard the paragraph
            if num_sentences != 0:
                paragraphs_dict[paragraph_id] = [doc_id, paragraph]
                paragraph_id += 1

        self.paragraphs_df = self.paragraphs_df.append(
            pd.DataFrame.from_dict(
                paragraphs_dict, orient="index", columns=["doc_id", "paragraph"]
            )
        )
        self.paragraphs_df.rename_axis("paragraph_id", axis="index", inplace=True)

    def __clean_content(self, text):
        """Make text lowercase, remove text in square brackets, remove punctuation, remove words containing numbers and other unwanted substrings."""

        url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
        email_regex = "[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}"
        # remove URLs
        text = re.sub(url_regex, "", text)
        # remove emails
        text = re.sub(email_regex, "", text)
        # remove numbers or "words" containing numbers
        # text = re.sub("(\w+\d\w*|\w*\d\w+)", "", text)
        text = re.sub("\w*\d\w*", "", text)
        # remove name initials (capital letter followed by a dot)
        text = re.sub("[A-Z]\.", "", text)
        # replace every non alphanumerical or whitespace or single quote character by a single space
        text = re.sub("[^a-zA-Z\d\s'’]+", " ", text)
        # replace every sequence of one ore more whitespace by a single space
        text = re.sub("\s+", " ", text)
        # => to lower case letters
        text = text.lower()

        return text

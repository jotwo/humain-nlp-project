import os
import re
import string
from io import StringIO

import pandas as pd
import spacy

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.layout import LAParams, LTTextBoxHorizontal
from pdfminer.converter import PDFPageAggregator
from pdfminer.pdfpage import PDFPage


class PDFCorpus:
    def __init__(self):

        self.nlp = spacy.load("en_core_web_sm")

        self.docs_df = pd.DataFrame(columns=["name"])
        self.docs_df.rename_axis("doc_id", axis="index", inplace=True)
        self.paragraphs_df = pd.DataFrame(columns=["content"])
        self.paragraphs_df.rename_axis("paragraph_id", axis="index", inplace=True)
        self.sentences_df = pd.DataFrame(columns=["content"])
        self.sentences_df.rename_axis("sentence_id", axis="index", inplace=True)

        self.corpus_df = pd.DataFrame(
            columns=["doc_id", "paragraph_id", "sentence_id", "token"]
        )
        self.corpus_df.rename_axis("token_id", axis="index", inplace=True)

    def add_multiple_pdfs(self, pdfs_dir_path):
        # we only extract pdf files from this directory
        pdf_names = [r for r in os.listdir(pdfs_dir_path) if r.endswith(".pdf")]
        for pdf_name in pdf_names:
            self.add_pdf(os.path.join(pdfs_dir_path, pdf_name))

    def add_pdf(self, pdf_filepath):

        # at this stage we are supposed to deal with '.pdf' files only
        # => 4 last characters are corresponding to '.pdf' and we remove them
        pdf_name = os.path.basename(pdf_filepath)[:-4]

        self.docs_df = self.docs_df.append({"name": pdf_name}, ignore_index=True)
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

    def _add_to_tables(self, pdf_id, paragraphs_list):

        if self.corpus_df.empty:
            paragraph_id = 0
            sentence_id = 0
            token_id = 0
        else:
            last_corpus_entry = self.corpus_df.iloc[-1]
            paragraph_id = last_corpus_entry["paragraph_id"] + 1
            sentence_id = last_corpus_entry["sentence_id"] + 1
            token_id = last_corpus_entry.name + 1

        corpus_dict = {}
        sentences_dict = {}
        paragraphs_dict = {}

        for paragraph in paragraphs_list:
            paragraph_doc = self.nlp(paragraph)

            last_sentence_id = sentence_id

            for sentence in paragraph_doc.sents:

                cleaned_sentence = self.__clean_content(sentence.text)

                # test if the cleaned sentence is not composed of whitespaces only
                if len(cleaned_sentence.strip()) != 0:

                    sentence_doc = self.nlp(cleaned_sentence)

                    for token in sentence_doc:
                        # we remove potential whitespaces around the token
                        # we are sure that the token is at least composed of one meaningful character at this point
                        if len(token.text.strip()) != 0:
                            # print(
                            #     f"doc_id: {doc_id},\tparagraph_id: {paragraph_id},\tsentence_id: {sentence_id},\t\ttoken_id: {token_id},\t\ttoken: {token.text}"
                            # )
                            corpus_dict[token_id] = [
                                pdf_id,
                                paragraph_id,
                                sentence_id,
                                token,
                            ]
                            token_id += 1

                    sentences_dict[sentence_id] = [sentence.text]

                    sentence_id += 1

            if sentence_id != last_sentence_id:
                paragraphs_dict[paragraph_id] = [paragraph]

                paragraph_id += 1

        self.corpus_df = self.corpus_df.append(
            pd.DataFrame.from_dict(
                corpus_dict,
                orient="index",
                columns=["doc_id", "paragraph_id", "sentence_id", "token"],
            )
        )
        self.corpus_df.rename_axis("token_id", axis="index", inplace=True)

        self.sentences_df = self.sentences_df.append(
            pd.DataFrame.from_dict(sentences_dict, orient="index", columns=["content"])
        )
        self.sentences_df.rename_axis("sentence_id", axis="index", inplace=True)

        self.paragraphs_df = self.paragraphs_df.append(
            pd.DataFrame.from_dict(paragraphs_dict, orient="index", columns=["content"])
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
        # replace special dashes by spaces
        text = re.sub("[–—]+", " ", text)
        # => to lower case letters
        text = text.lower()
        # remove punctuation
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
        # remove numbers or "words" containing numbers
        text = re.sub("\w*\d\w*", "", text)
        # remove greek characters
        text = re.sub("[α-ωΑ-Ω]+", "", text)
        # remove some remaining parts of math expressions
        text = re.sub("ˆ[A-Za-z]*", "", text)
        # remove different types of bullet
        text = re.sub("[\u2022\u2023\u25E6\u2043\u2219]", "", text)
        # replace whitespace characters (newlines, tabs, ...) with space
        text = re.sub("\s", " ", text)
        # remove additional unwanted characters
        text = re.sub("[«»“”©●…↓]", "", text)

        return text

    def get_corpus_df(self):
        return self.corpus_df.copy()

    def get_sentences_df(self):
        return self.sentences_df.copy()

    def get_paragraphs_df(self):
        return self.paragraphs_df.copy()

    def get_docs_df(self):
        return self.docs_df.copy()
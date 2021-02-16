import pandas as pd

from preprocessing import PDFCorpus

pdf_corpus = PDFCorpus()


pdf_corpus.add_multiple_pdfs(
    "/home/jo/becode/projects/Project 7/humain-nlp-project/data/reports/"
)


pdf_corpus.get_tokens_df()

pdf_corpus.get_tokens_df().to_pickle(
    "/home/jo/becode/projects/Project 7/humain-nlp-project/src/pickle/tokens.pkl"
)
pdf_corpus.get_sentences_df().to_pickle(
    "/home/jo/becode/projects/Project 7/humain-nlp-project/src/pickle/sentences.pkl"
)
pdf_corpus.get_paragraphs_df().to_pickle(
    "/home/jo/becode/projects/Project 7/humain-nlp-project/src/pickle/paragraphs.pkl"
)
pdf_corpus.get_docs_df().to_pickle(
    "/home/jo/becode/projects/Project 7/humain-nlp-project/src/pickle/docs.pkl"
)

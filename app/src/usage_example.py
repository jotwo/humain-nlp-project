from preprocessing import PDFCorpus


pdf_corpus = PDFCorpus()

pdf_corpus.add_multiple_pdfs("../../data/reports/")

pdf_corpus.get_tokens_df()

pdf_corpus.get_docs_df().to_pickle("../../pickles/docs.pkl")
pdf_corpus.get_paragraphs_df().to_pickle("../../pickles/paragraphs.pkl")
pdf_corpus.get_sentences_df().to_pickle("../../pickles/sentences.pkl")
pdf_corpus.get_tokens_df().to_pickle("../../pickles/tokens.pkl")

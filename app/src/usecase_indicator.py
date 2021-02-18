"""This unction implements the usecase model for classification of sentences"""
from tensorflow import keras
from transformers import BertTokenizer, BertConfig, TFBertModel
import numpy as np
import pandas as pd


def usecase_indicator(corpus, n_last, model: str, quality: float = 1.4):
    """Calculates a usecase score for each new sentence in pdf_corpus.get_sentences_df()
    and filters the values scoring higher than the required quality.
    Then, only the sentence with the max score in the paragraph is returned.
    Output is a dataframe including full paragraph text.
    ::corpus:: pdf_corpus object
    ::n_last:: only the last n docs of the corpus will be postprocessed
    ::model:: model.h5 path
    ::quality:: higher values produce less results (max ~1.6)"""

    # predict based on the pretrained bert tokenizer and our finetuned classifier
    config = BertConfig.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", config=config)

    new_doc_ids = corpus.get_docs_df().index[n_last * -1 :]
    new_sentences = corpus.get_sentences_df().loc[
        corpus.get_sentences_df().doc_id.isin(new_doc_ids)
    ]

    encoded_data = tokenizer(
        list(new_sentences.sentence),
        max_length=40,
        truncation=True,
        add_special_tokens=True,
        return_tensors="tf",
        padding="max_length",
        return_attention_mask=True,
        return_token_type_ids=False,
    )

    model = keras.models.load_model(model)

    y = model.predict(
        x={
            "input_ids": encoded_data["input_ids"],
            "attention_mask": encoded_data["attention_mask"],
        }
    )

    new_sentences.loc[:, "usecase_score"] = y["usecase"].ravel()
    sentences_df_filtered = new_sentences.loc[new_sentences["usecase_score"] > 1.4]

    idx = (
        sentences_df_filtered.groupby(["paragraph_id"])["usecase_score"].transform(max)
        == sentences_df_filtered["usecase_score"]
    )

    # join full paragraph text
    result = sentences_df_filtered[idx].merge(
        corpus.get_paragraphs_df()["paragraph"],
        how="inner",
        on="paragraph_id",
        copy=False,
        validate="one_to_one",
    )

    return result


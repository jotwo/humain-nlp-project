"""This unction implements the usecase model for classification of sentences"""
import sys
from tensorflow import keras
from transformers import BertTokenizer, BertConfig, TFBertModel
import numpy as np
import pandas as pd
from download_model import download_model
from qna import qa


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
    thresholded_sentences = new_sentences.loc[new_sentences[
      "usecase_score"] > quality]

    thresholded_paragraph_ids = thresholded_sentences.paragraph_id.\
      unique().tolist()



    # concatenate full paragraph text for thresholded paragraph_ids
    thresholded_paragraph_text = new_sentences.loc[
      new_sentences.paragraph_id.isin(thresholded_paragraph_ids),
      ['paragraph_id', 'sentence']].groupby(by = 'paragraph_id')[
        'sentence'].apply(lambda x: '. '.join(x))

    thresholded_paragraph_text.rename('paragraph', inplace=True)

    maxed_sentences_idx = (
        thresholded_sentences.groupby(["paragraph_id"])["usecase_score"].transform(max)
        == thresholded_sentences["usecase_score"]
    )

    # join full paragraph text
    result = thresholded_sentences[maxed_sentences_idx].merge(
        thresholded_paragraph_text,
        how="inner",
        left_on="paragraph_id",
        right_index = True,
        copy=False,
        validate="one_to_one",
    ).reset_index(drop = True)

    print('Classification done')
    sys.stdout.flush()

    # heroku might have again deleted the QnA model because it's so big
    download_model()

    detailed_df = qa(result)
    print('QnA done')
    sys.stdout.flush()

    return detailed_df


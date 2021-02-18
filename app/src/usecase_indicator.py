import tensorflow_addons as tfa
from tensorflow import keras
from transformers import BertTokenizer, BertConfig
from transformers import TFBertModel
import numpy as np
import pandas as pd

def usecase_indicator(sentences_df, model: str, quality: float = 3.5):
    """Calculates a usecase score for each sentence in pdf_corpus.get_sentences_df()
    and filters the values scoring higher than the required quality.
    Then, only the sentence with the max score in the paragraph is returned.
    Output is a dataframe.
    ::model:: model.h5 path
    ::quality:: higher values produce less results (max ~1.6)"""
    
    # predict based on the pretrained bert tokenizer and our finetuned classifier
    config = BertConfig.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', config = config)
    encoded_data = tokenizer(list(sentences_df.sentence),
                           max_length = 40,
                           truncation = True,
                           add_special_tokens = True,
                           return_tensors = 'tf',
                           padding = 'max_length',
                           return_attention_mask = True,
                           return_token_type_ids = False)
    
    model = keras.models.load_model(model)
    
    y = model.predict(x={'input_ids': encoded_data['input_ids'],
                       'attention_mask': encoded_data['attention_mask']})
    
    sentences_df.loc[:, 'usecase_score'] = y['usecase'].ravel()
    sentences_df_filtered = sentences_df.loc[sentences_df['usecase_score'] > 1.4]
    
    idx = sentences_df_filtered.groupby(['paragraph_id'])['usecase_score'].\
                  transform(max) == sentences_df_filtered['usecase_score']

    # join full paragraph text
    result = sentences_df_filtered[idx].merge(pdf_corpus.get_paragraphs_df()['paragraph'],
                                              how='inner',
                                              on='paragraph_id',
                                              copy=False,
                                              validate='one_to_one')    
    
    return result
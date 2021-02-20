"""This file defines the function for downloading our finetuned BERT
classifier model and the pretrained Bert QnA model"""

import requests
import sys
from pathlib import Path
from transformers import BertForQuestionAnswering


def download_model():
    if (
        not Path("model_downloaded").is_file()
        or not Path("usecase_indicator.h5").is_file()
    ):

        url = "https://b0ykepubbucket.s3-eu-west-1.amazonaws.com/usecase_indicator.h5"
        r = requests.get(url, stream=True)
        chunk_progress = 0
        with open("usecase_indicator.h5", "wb") as modelfile:
            for chunk in r.iter_content(chunk_size=8388608):
                if chunk:
                    modelfile.write(chunk)
                    chunk_progress += 1
                    print(f"Downloading model 1/2 in background: {chunk_progress*8}MB")
                    sys.stdout.flush()
            else:
                open("model_downloaded", "w").close()
    
    if (
        not Path("modelqna_downloaded").is_file()
        or not Path("./BertLSquad/pytorch_model.bin").is_file()
    ):
        print('Started model 2/2 download in background')
        sys.stdout.flush()
        model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        model.save_pretrained("./BertLSquad")
        open("modelqna_downloaded", "w").close()
        print('Model 2/2 download completed')
        sys.stdout.flush()

    return


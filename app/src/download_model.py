"""This file defines the function for downloading our finetuned BERT model"""
import requests
import sys
from pathlib import Path

def download_model():
    if not Path('model_downloaded').is_file() or\
        not Path('usecase_indicator.h5').is_file():

        url = 'https://b0ykepubbucket.s3-eu-west-1.amazonaws.com/usecase_indicator.h5'
        r = requests.get(url, stream = True)
        chunk_progress = 0
        with open("usecase_indicator.h5", "wb") as modelfile:
            for chunk in r.iter_content(chunk_size = 8388608):
                if chunk:
                    modelfile.write(chunk)
                    chunk_progress += 1
                    print(f"Downloaded: {chunk_progress*8}MB\n")
                    sys.stdout.flush()
            else:
                open('model_downloaded','w').close()
    return

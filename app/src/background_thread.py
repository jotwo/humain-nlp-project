"""this file holds the background thread code for processing
the uploads"""
import os
import sys
from werkzeug.utils import secure_filename
from usecase_indicator import usecase_indicator
from download_model import download_model
import threading
from pickle import dump

def process(upload_folder, corpus, files):

    # (re)download the models if necessary in another background thread
    # this job highly likely finishes before prediction starts
    threading.Thread(target=download_model).start()

    # we only want to interpret the newly uploaded files
    # initiate the counter
    new_uploads = 0

    for file in files:

        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_folder, filename)
        corpus.add_pdf(filepath)

        new_uploads += 1

        print(corpus.get_docs_df().iloc[-1].to_string() + 'was preprocessed')
        # flushing the output buffer makes the print message available
        # on heroku log
        sys.stdout.flush()

    # after all files are added to the corpus we can start postprocessing
    # the usecase_indicator function was refactored to inlude the
    # second QnA process
    detailed_df = usecase_indicator(corpus = corpus,
    								n_last = new_uploads,
    								model = 'usecase_indicator.h5',
    								quality = 1.45)

    # save the dataframe as pickle
    with open('detailed_df.pkl', 'wb') as f:
    	dump(detailed_df, f)

    return
import os
import pickle
import pandas as pd

import sys
import threading

from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename

from preprocessing import PDFCorpus
from usecase_indicator import usecase_indicator
from download_model import download_model

import spacy
from spacy import displacy

# support for downloading the model in the background
# while preprocessing
threading.Thread(target=download_model).start()

app = Flask(__name__)

app.secret_key = "secret key"  # for encrypting the session
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {"pdf"}

UPLOAD_FOLDER = "../uploads"

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

pdf_corpus = PDFCorpus()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def upload_form():
    return render_template("upload.html")


@app.route("/", methods=["POST"])
def upload_file():
    if request.method == "POST":
        if "files[]" not in request.files:
            flash("no file part")
            return redirect(request.url)

        files = request.files.getlist("files[]")

        for file in files:

            if file and allowed_file(file.filename):

                filename = secure_filename(file.filename)

                file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

                pdf_corpus.add_pdf(os.path.join(UPLOAD_FOLDER, filename))

                # # no need to create copies of the dataframes I think
                # docs_df = pdf_corpus.get_docs_df().copy()
                # paragraphs_df = pdf_corpus.get_paragraphs_df().copy()
                # sentences_df = pdf_corpus.get_sentences_df().copy()
                # tokens_df = pdf_corpus.get_tokens_df().copy()

                print(pdf_corpus.get_docs_df())
                # flushing the output buffer makes the print message available
                # on heroku log
                sys.stdout.flush()
                # print(paragraphs_df)
                # print(sentences_df)
                # print(tokens_df)

            else:
                flash("Only PDF file(s) supported")
                break

        # when preprocessing is successful
        else:
            flash("File(s) successfully processed")

        # after all files are added to the corpus we can start postprocessing
        # first we only select paragraphs with usecase sentences
        usecase_indication = usecase_indicator(pdf_corpus.get_sentences_df(),
                                                'usecase_indicator.h5',
                                                quality = 1.4)

        # then we apply QnA to the selected paragraphs
        

        return redirect("/")


@app.route("/text")
def text():
    return render_template("text_extractor.html")


@app.route("/process", methods=["POST"])
def text_processing():
    if request.method == "POST":
        # with open('E:/BeCodeProjects/HumAIn_Project/text0.txt','r',encoding="utf8") as f:
        # return render_template("text_extractor.html",text=f.read())
        choice = request.form.get("taskoption")

        rawtext = request.form.get("rawtext")

        print(rawtext)
        doc = nlp(rawtext)
        # print(doc)
        d = []
        for sent in doc.sents:
            # print(ent)
            d.append((sent.label_, sent.text))
            df = pd.DataFrame(d, columns=("industry", "output"))
            print(df)
            test = df["output"]
            print(test)

        if choice == "Marketing":
            results = test
            print(results)
            num_of_results = len(results)
        """    
        elif choice == 'Sales':
            results = PERSON_named_entity
            num_of_results = len(results)
        elif choice == 'IT':
            results = GPE_named_entity
            num_of_results = len(results)
        elif choice == 'Data':
            results = MONEY_named_entity
            num_of_results = len(results)
        """
    return render_template(
        "text_extractor.html", results=results, num_of_results=num_of_results
    )

if __name__ == "__main__":
    # You want to put the value of the env variable PORT if it exist
    # (some services only open specifiques ports)
    port = int(os.environ.get('PORT', 5000))
    # Threaded option to enable multiple instances for
    # multiple user access support
    # You will also define the host to "0.0.0.0" because localhost
    # will only be reachable from inside de server.
    app.run(host="0.0.0.0", threaded=True, debug=True, port=port)

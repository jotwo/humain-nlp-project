import os
import pickle
import pandas as pd

from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename

from preprocessing import PDFCorpus

import spacy
from spacy import displacy

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

                docs_df = pdf_corpus.get_docs_df().copy()
                paragraphs_df = pdf_corpus.get_paragraphs_df().copy()
                sentences_df = pdf_corpus.get_sentences_df().copy()
                tokens_df = pdf_corpus.get_tokens_df().copy()

                print(docs_df)
                # print(paragraphs_df)
                # print(sentences_df)
                # print(tokens_df)

                flash("File(s) successfully uploaded")

            else:
                flash("Upload PDF file(s)")

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
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=True)

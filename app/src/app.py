import os
import sys
import threading
from pickle import load

import pandas as pd
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename

from download_model import download_model
from background_thread import process
from preprocessing import PDFCorpus

# support for downloading the model in the background
# while preprocessing
threading.Thread(target=download_model).start()

app = Flask(__name__)

app.secret_key = "secret key"  # for encrypting the session
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

UPLOAD_FOLDER = "../uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {"pdf"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# seting the maximum column width to display all columns
pd.set_option("display.max_colwidth", None)


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

            else:
                flash("Only PDF file(s) are supported")
                return

        pdf_corpus = PDFCorpus()
        nlp = pdf_corpus.nlp

        # strat processing function as a thread
        threading.Thread(target=process, args = (
                                    app.config["UPLOAD_FOLDER"],
                                    pdf_corpus,
                                    files)).start()

        # in the end we flash the result is ready and show the button
        flash(f'The result should be available in 15 mins')

        return redirect("/")


@app.route("/text", methods=["POST"])
def text():
    if request.method == "POST":
        # make detailed_df available globally
        global detailed_df
        with open('detailed_df.pkl', 'rb') as f:
            detailed_df = load(f)
        # global paragraphs_df
        global text
        text = detailed_df["paragraph"].to_string(index = False)

        return render_template("text_extractor.html", text=text)


@app.route("/process", methods=["POST"])
def text_processing():
    if request.method == "POST":
        global detailed_df
        global text, results, num_of_results

        choice1 = request.form.get("taskoption")  # Function
        choice2 = request.form.get("taskoption2")  # Industry

        exhibit_ind = detailed_df.loc[detailed_df["industry"] == "exhibit"]["usecase"]
        exhibit_fn = detailed_df.loc[detailed_df["function"] == "exhibit"]["usecase"]
        dr_ind = detailed_df.loc[detailed_df["industry"] == "data richness"]["usecase"]
        dr_fn = detailed_df.loc[detailed_df["function"] == "data richness"]["usecase"]
        pa_ind = detailed_df.loc[detailed_df["industry"] == "predictive analytics"][
            "usecase"
        ]
        pa_fn = detailed_df.loc[detailed_df["function"] == "productivity and growth"][
            "usecase"
        ]
        if choice1 == "exhibit" or choice2 == "exhibit":
            results = exhibit_ind
            text = detailed_df["paragraph"]

            num_of_results = len(results)
        if choice1 == "data richness" or choice2 == "data richness":
            results = dr_ind
            text = detailed_df["paragraph"]
            num_of_results = len(results)
        if choice1 == "predictive analytics" or choice2 == "productivity and growth":
            results = pa_ind
            text = detailed_df["paragraph"]
            num_of_results = len(results)

    return render_template(
        "text_extractor.html", results=results, num_of_results=num_of_results
    )

@app.route("/files")
def display_files():
    global detailed_df
    global text, results, num_of_results
    text = text
    return render_template(
        "text_extractor.html", results=results, num_of_results=num_of_results, text=text
    )


if __name__ == "__main__":
    # You want to put the value of the env variable PORT if it exist
    # (some services only open specifiques ports)
    port = int(os.environ.get('PORT', 5000))
    # Threaded option to enable multiple instances for
    # multiple user access support
    # You will also define the host to "0.0.0.0" because localhost
    # will only be reachable from inside de server.
    app.run(host="0.0.0.0", threaded=True, debug=False, port=port)

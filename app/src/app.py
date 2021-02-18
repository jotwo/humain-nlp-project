import os
import sys
import threading
import pandas as pd
from flask import Flask, request, flash, redirect, render_template
from werkzeug.utils import secure_filename

from preprocessing import PDFCorpus
from usecase_indicator import usecase_indicator
from download_model import download_model
from qna import qa

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

# seting the maximum column width to display all columns
pd.set_option("display.max_colwidth", None)

pdf_corpus = PDFCorpus()

nlp = pdf_corpus.nlp


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def upload_form():
    return render_template("upload.html")


@app.route("/", methods=["POST"])
def upload_file():

    # the postprocessed result dataframe from Orhan will be available
    # to other routes
    global results_df

    # heroku might delete one of the model files while our dyno is running
    # download it again if needed when executing an upload
    threading.Thread(target=download_model).start()

    if request.method == "POST":
        if "files[]" not in request.files:
            flash("no file part")
            return redirect(request.url)

        files = request.files.getlist("files[]")

        # we only want to interpret the newly uploaded files
        # initiate the counter
        new_uploads = 0

        for file in files:

            if file and allowed_file(file.filename):

                filename = secure_filename(file.filename)
                global detailed_df

                file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
                pdf_corpus.add_pdf(os.path.join(UPLOAD_FOLDER, filename))

                new_uploads += 1

                print(pdf_corpus.get_docs_df().iloc[-1].to_string())
                # flushing the output buffer makes the print message available
                # on heroku log
                sys.stdout.flush()


                flash("File(s) successfully uploaded")
            else:
                flash("Only PDF file(s) are supported")
                break

        # after all files are added to the corpus we can start postprocessing
        # first we only select paragraphs with usecase sentences in newly
        # uploaded files
        usecase_indication = usecase_indicator(corpus = pdf_corpus,
                                                n_last = new_uploads,
                                                model = 'usecase_indicator.h5',
                                                quality = 1.4)
        
        # Show the top 10 results of the classification in the console
        print(f'{len(usecase_indication)} usecases found')
        sys.stdout.flush()
        counter = 0
        for i, row in usecase_indication.iterrows():
            if counter < 10:
                print(row['sentence'])
                sys.stdout.flush()
                counter += 1
            else:
                break

        # then we apply QnA to the selected paragraphs


        detailed_df = qa(usecase_indication)


        # in the end we flash the result is ready and show the button
        flash(f'Text interpretation finished')

        return redirect("/")


@app.route("/text", methods=["POST"])
def text():
    if request.method == "POST":
        global detailed_df
        # global paragraphs_df
        global text
        text = detailed_df["paragraph"]

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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)

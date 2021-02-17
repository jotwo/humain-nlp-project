import os
from flask import Flask, flash, request, redirect, render_template, session
from werkzeug.utils import secure_filename
import re
import pandas as pd
import spacy
from spacy import displacy

nlp = spacy.load('en_core_web_sm')
from src.preprocessing import PDFCorpus
import pickle

pdf_corpus = PDFCorpus()

# para_model=pickle.load('E:/BeCodeProjects/HumAIn_Project/src/pickle/paragraphs.pkl')


app = Flask(__name__)

app.secret_key = 'secret key'  # for encrypting the session
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

path = os.getcwd()
UPLOAD_FOLDER = os.path.join(path, '../uploads')
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'csv', 'json'])
# seting the maximum column width to display all columns
pd.set_option('display.max_colwidth', -1)


def allowed_file(filename):
    # if filename.rsplit('.', 1)[1].lower() == 'json':
    # parse_json_file(filename)
    # if filename.rsplit('.', 1)[1].lower() == 'csv':
    # parse_csv_file(filename)

    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'files[]' not in request.files:
            flash('no file part')
            return redirect(request.url)

        files = request.files.getlist('files[]')

        for file in files:

            if file and allowed_file(file.filename):

                filename = secure_filename(file.filename)
                global detailed_df
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                pdf_corpus.add_pdf(os.path.join(UPLOAD_FOLDER, filename))
                tab1 = []
                docs_df = pdf_corpus.get_docs_df().copy()
                paragraphs_df = pdf_corpus.get_paragraphs_df().copy()
                sentences_df = pdf_corpus.get_sentences_df().copy()
                tokens_df = pdf_corpus.get_tokens_df().copy()

                detailed_df = tokens_df[['token']].copy()
                detailed_df['doc_name'] = tokens_df['doc_id'].apply(lambda x: docs_df.loc[x, 'name'])
                detailed_df['paragraph'] = tokens_df['paragraph_id'].apply(lambda x: paragraphs_df.loc[x, 'paragraph'])
                detailed_df['sentence'] = tokens_df['sentence_id'].apply(lambda x: sentences_df.loc[x, 'sentence'])

                usecase_df = pd.read_csv('E:/BeCodeProjects/HumAIn_Project/src/most_likely_usecase_per_paragraph.csv')

                detailed_df['industry'] = usecase_df['industry']  # .apply(lambda x: usecase_df.loc[x, 'industry'])
                detailed_df['usecase'] = usecase_df['usecase']  # .apply(lambda x: usecase_df.loc[x, 'usecase'])
                detailed_df = detailed_df[['doc_name', 'paragraph', 'sentence', 'token', 'industry', 'usecase']]

                # print(detailed_df.columns)

                flash('File(s) successfully uploaded')


            else:

                flash('Upload file(s) in pdf format')

        return redirect('/')


@app.route('/text', methods=['POST'])
def text():
    if request.method == 'POST':
        global detailed_df
        text = detailed_df['paragraph']
        # print(detailed_df)
        return render_template('text_extractor.html', text=text)


@app.route('/process', methods=['POST'])
def text_processing():
    if request.method == 'POST':
        # with open('E:/BeCodeProjects/HumAIn_Project/text0.txt','r',encoding="utf8") as f:
        # return render_template("text_extractor.html",text=f.read())
        global detailed_df
        choice1 = request.form.get('taskoption')  # Function
        choice2 = request.form.get('taskoption2')  # Industry
        print(detailed_df)
        """
        rawtext = detailed_df.copy() # request.form.get('rawtext')
        for raw_text in rawtext:
            print(raw_text)
            doc = nlp(raw_text)
        # print(doc)
            d = []
            for sent in doc.sents:
            # print(ent)
                d.append((sent.label_, sent.text))
                df = pd.DataFrame(d, columns=('industry', 'output'))
            
                test = df['output']

            if choice1 == 'Marketing' :
                results = test
                print(results)
                num_of_results = len(results)
            """
    return render_template("text_extractor.html", results=results, num_of_results=num_of_results)


if __name__ == '__main__':
    app.run(debug=True, threaded=True, port=5000)

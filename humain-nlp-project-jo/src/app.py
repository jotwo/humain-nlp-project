import os
from flask import Flask, flash, request, redirect, render_template
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

"""
def parse_json_file(file):
    col_names=['function','industry']
    csv_data=pd.read_json(file,names=col_names,header=None)
    for i,row in csv_data.iterrows():
        print(i,row['function'],row['industry'])
def parse_csv_file(file):
"""


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

                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                flash('File(s) successfully uploaded')

            else:
                flash('Upload file(s) in pdf format')

        return redirect('/')

@app.route('/text')
def text():
    return render_template('text_extractor.html')
@app.route('/process', methods=['POST'])
def text_processing():
    if request.method == 'POST':
        # with open('E:/BeCodeProjects/HumAIn_Project/text0.txt','r',encoding="utf8") as f:
        # return render_template("text_extractor.html",text=f.read())
        choice = request.form.get('taskoption')

        rawtext = request.form.get('rawtext')

        print(rawtext)
        doc = nlp(rawtext)
        #print(doc)
        d = []
        for sent in doc.sents:
            #print(ent)
            d.append((sent.label_, sent.text))
            df = pd.DataFrame(d, columns=('industry', 'output'))
            print(df)
            test = df['output']
            print(test)

        if choice == 'Marketing':
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
    return render_template("text_extractor.html", results=results, num_of_results=num_of_results)


if __name__ == '__main__':
    app.run(debug=True, threaded=True, port=5000)

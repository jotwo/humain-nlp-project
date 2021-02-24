# HumAIn NLP Project - An AI Knowledge platform

## Introduction

The applications that publishes articles on daily news, entertainment, sports demands for the platform that summarizes the text. Because of the busy schedule users prefer to read the sumary of the articles before deciding to read entire document/article. Reading a summary helps us to identify the relevant area of information that gives brief context of the story. Thus, summarization is the most essential application under natural language processing (NLP) that defines a task of producing a concise and fluent summary while preserving key information and semantic understanding.

Our service is deployed on below link, you may get an Application error at first because our instance is waking from sleep.  
#### [aaik.herokuapp.com](http://aaik.herokuapp.com)

## Problem Definition
1. Not enough time to read AI industry reports
2. hard to digest
3. Slow to go back and look up
4. Want to have use cases at fingertips

## Mission
* Produce shorter version of a source files 
* Preserve the meaning and key contents
* Digest large amounts of text
* Information filtering and readability assessment
* Atomization and simplification of applications and companies operations

## Input/Output Flow
<img src="https://github.com/jotwo/humain-nlp-project/blob/manasa2/humain-nlp-project-jo/src/HumAInFlowDiagram.jpg" alt="alt text" width=500 height=300>

The overall system works by uploading the pdf file(s) and generates the summarized text as paragraph by passing though processing steps. First, the system allows to upload the files, if the uploaded file is pdf then a message will be dispalyed saying <tt>"File(s) uploaded successfully"</tt> and <tt>"Text interpretation finished"</tt> along with the button for <i>"check content"</i> in the second page for summarized text. If the uploaded file is not pdf then it displays <tt>"Only PDF file(s) are supported"</tt>. Upon clicking the "check content" button, we go to second page to select the filters to view relevant paragraphs from the pdf documents. We have provided two <b>filters</b>- 1. industry 2. function. With these two filters, we actually querying a system to generate relevant usecase as links. User can select the interested link to view the paragraph/information. 

## Modules
1. Upload pdf file(s)
2. Preprocessing
3. Classifying the sentences  
      
    The goal here is to find the use case highlights for presentation to the user, and to point out paragrapshs that the next Question
    Answering component should focus on.  
    We built a dataset of 450 sentences by selection from the provided data, half of which true use cases. Intuitively and generally
    they were mostly a formulation of some type of action that contributed to a business value, and we set out to model this by using
    feature extraction from the BERT uncased model. Ater finetuning on 80% of the data, our classification reached 95% validation
    accuracy.
5. Building QA system
6. Building an App

## Team
* Joachim Kotek
* Philippe Fimmers
* Orhan Nurkan
* Manasa Noolu








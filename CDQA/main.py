#pip install torch===1.2.0 -f https://download.pytorch.org/whl/torch_stable.html

import pandas as pd
from ast import literal_eval

from cdqa.utils.filters import filter_paragraphs
from cdqa.utils.download import download_model, download_bnpp_data
from cdqa.pipeline.cdqa_sklearn import QAPipeline
from cdqa.utils.converters import pdf_converter

# Download data and models
'''
download_bnpp_data(dir='./data/bnpp_newsroom_v1.1/')
download_model(model='bert-squad_1.1', dir='./models')
'''

df = pdf_converter(directory_path='./data/pdf')
#print(df)
'''
# Loading data and filtering / preprocessing the documents
df = pd.read_csv('data/bnpp_newsroom_v1.1/bnpp_newsroom-v1.1.csv', converters={'paragraphs': literal_eval})
'''

df = filter_paragraphs(df)

# Loading QAPipeline with CPU version of BERT Reader pretrained on SQuAD 1.1
cdqa_pipeline = QAPipeline(reader='models/bert_qa.joblib')

# Fitting the retriever to the list of documents in the dataframe
cdqa_pipeline.fit_retriever(df=df)

# Sending a question to the pipeline and getting prediction
#query = 'Since when does the Excellence Program of BNP Paribas exist?'
#query = 'Who provides support to young graduates?'
#query = 'How do you call a Javascirpt function?'
#query = 'What is the URL for localhost DirectAccess?'
#query = 'What is the Matthew Principle?'
query = 'Where is there a piped command uglify?'
N = 2
prediction = cdqa_pipeline.predict(query=query,n_predictions=N)

#print(prediction)

print('query: {}\n'.format(query))
print('answer: {}\n'.format(prediction[0]))
print('title: {}\n'.format(prediction[1]))
#print('paragraph: {}\n'.format(prediction[2]))

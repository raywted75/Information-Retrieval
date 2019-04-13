import os
import re
import jieba
import pandas as pd
import argparse
import xml.etree.ElementTree as ET
from param import *


# set dictionary to traditional Chinese
jieba.set_dictionary('data/jieba/dict.txt.big')

# load stopwords
stopwords=[]
with open('data/jieba/stopwords.txt', 'r', encoding='UTF-8') as file:
    for data in file.readlines():
        data = data.strip()
        stopwords.append(data)


# Functions
def process_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', dest='rocchio_feedback', action='store_true', required=False)
    parser.add_argument('-i', dest='query_file', default='data/queries/query-test.xml', required=True)
    parser.add_argument('-o', dest='ranked_list', default='search_result.csv', required=True)
    parser.add_argument('-m', dest='model_dir', default='data/model', required=True)
    parser.add_argument('-d', dest='ntcir_dir', default='data/CIRB010', required=True)
    
    return parser.parse_args()


def segment(text):
    if not text:
        text = ' '
    
    words = ''
    if cut_type == 0:
        words = jieba.cut(text)
    elif cut_type == 1:
        words = jieba.cut(text, cut_all=True)
    elif cut_type == 2:
        words = jieba.cut_for_search(text)
    
    if use_stopwords:
        words = list(filter(lambda w: w not in stopwords and w != '\n', words))
    
    return ' '.join(words) + ' '


def doc2dict(file):
    root = ET.parse(file).getroot()
    text = ''
    for paragraph in root.find('doc/text').iter():
        text += paragraph.text

    dict_of_doc = {
        'id': root.find('doc/id').text, 
        'title': segment(root.find('doc/title').text), 
        'date': root.find('doc/date').text, 
        'text': segment(text)
    }

    return dict_of_doc


def query2dicts(file):
    root = ET.parse(file).getroot()
    text = ''
    query_list = []
    for topic in root.findall('topic'):
        dict_of_query = {
            'number': topic.find('number').text,
            'title': segment(topic.find('title').text),
            'question': segment(topic.find('question').text),
            'concepts': segment(topic.find('concepts').text),
            'narrative': segment(topic.find('narrative').text)
        }
        query_list.append(dict_of_query)

    return query_list


def create_doc(doc_dir, model_dir):
    doc_list = []
    all_file_list = os.path.join(model_dir, 'file-list')
    with open(all_file_list) as file_list:
        for doc in file_list:
            doc_path = os.path.join(doc_dir, doc[8:])
            doc_list.append(doc2dict(doc_path.rstrip('\n')))

    doc = pd.DataFrame(doc_list)
    # doc.to_csv('doc.csv')
    
    return doc


def create_query(query_file):
    query_list = query2dicts(query_file)
    query = pd.DataFrame(query_list)
    # query.to_csv('query.csv')

    return query


def create_corpus(doc, query):
    doc_corpus = doc.title + doc.text
    query_corpus = query.concepts + query.title
    corpus = pd.concat((doc_corpus, query_corpus)).values

    return corpus


def save_result(result, ranked_list):
    search_result = pd.DataFrame(result , columns=['query_id', 'retrieved_docs'])
    search_result.to_csv(ranked_list, index=False)

    return None
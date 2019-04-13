import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from param import *
from utils import *


def calculate_okapi(corpus):
    dictionary = {}
    term_index = 0
    df = np.array([])
    raw_tf = []
    dl = []
    for doc_i in range(len(corpus)):
        tf = {}
        dl.append(len(corpus[doc_i]))
        for term in corpus[doc_i].split():
            if term not in dictionary:
                dictionary[term] = term_index           
                df = np.append(df, [0])
                tf[term_index] = 1
                term_index += 1
            else:
                if dictionary[term] in tf:
                    tf[dictionary[term]] += 1
                else:
                    tf[dictionary[term]] = 1
        
        for term_i in tf.keys():
            df[term_i] += 1
            
        for term_j, term_freq in tf.items():
            raw_tf.append((doc_i, term_j, term_freq))

    N = len(corpus)
    avdl = sum(dl) / N
    idf = np.log((N - df + 0.5) / (df + 0.5))
    okapi_tf = [0] * len(raw_tf)

    for i, (doc_i, term_j, term_freq) in enumerate(raw_tf):
        tf_idf = (k1 + 1) * term_freq / (k1 * (1 - b + b * dl[doc_i] / avdl) + term_freq) * idf[term_j]
        okapi_tf[i] = (doc_i, term_j, tf_idf)

    okapi_tf = np.array(okapi_tf).T
    row = okapi_tf[0].astype(int)
    col = okapi_tf[1].astype(int)
    weight = okapi_tf[2]
    vectors = csr_matrix((weight, (row, col)))
    
    return vectors


def calculate_ranking(query, doc, vectors, rocchio_feedback):
    query_id = []
    retrieved_docs = []
    top_docs = []
    for i in range(len(query)):
        query_vector = vectors[-1*len(query) + i]
        sim = cosine_similarity(vectors, query_vector).T[0]
        top_doc_index = sim.argsort()[::-1][1:101]
        if rocchio_feedback:
            for _ in range(feedback_times):
                related_doc_num = min(len(sim[sim > threshold]), max_related)
                query_vector += b_r / related_doc_num * np.sum(vectors[top_doc_index[:related_doc_num]], axis=0)
                sim = cosine_similarity(vectors, query_vector).T[0]
                top_doc_index = sim.argsort()[::-1][1:101]

        top_docs = doc.id[top_doc_index].fillna(' ').values
        query_id.append(i+11)
        retrieved_docs.append(' '.join(top_docs))

    query_id = np.array([query_id]).T
    retrieved_docs = np.array([retrieved_docs]).T
    result = np.concatenate((query_id, retrieved_docs), axis=1)

    return result


def main():

    args = process_command()

    print("Creating documents...")
    doc = create_doc(args.ntcir_dir, args.model_dir)

    print("Creating queries...")
    query = create_query(args.query_file)

    print("Creating corpus...")
    corpus = create_corpus(doc, query)

    print("Calculating Okapi weighting...")
    vectors = calculate_okapi(corpus)

    print("Calculating ranking...")
    result = calculate_ranking(query, doc, vectors, args.rocchio_feedback)

    save_result(result, args.ranked_list)


if __name__ == "__main__":
    main()
import argparse
import pickle
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.neighbors import kneighbors_graph as knng
from scipy.sparse import save_npz
with open('data/paragraphs.data','rb') as f :
    doc_model = pickle.load(f )
with open('data/paragraphs.data.docvecs.vectors_docs.npy','rb') as f_arr :
    doc_array = np.load(f_arr )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', help='calculate connection', action='store_const', const='connectivity')
    args = parser.parse_args()
    X = normalize(doc_array)
    if args.c:
        result = knng(X, n_neighbors=10, n_jobs=-1, mode=args.c)
        save_npz("graph_connect", result)
    else:
        result = knng(X, n_neighbors=10, n_jobs=-1, mode='distance')
        save_npz("graph_distance", result)


from gensim.models.doc2vec import Doc2VecKeyedVectors
import pickle
from typing import List
from collections import namedtuple
import nltk
from gensim.models import Doc2Vec 
from gensim.models.doc2vec import TaggedDocument 
from pages_generator import Page
from multiprocessing import Pool
import string
import re 
from itertools import chain, islice
import pickle
from gensim.models import Doc2Vec 

stopwords = set(nltk.corpus.stopwords.words('english'))
quotes =set(["``","''"])
forbidden  = stopwords.union( quotes).union(set(string.punctuation))

def clean_text(article):
    words = filter(lambda x: x not in forbidden, nltk.word_tokenize(article.text.lower()))
    words = list(map(lambda x: re.sub("\d+(th|nd|st)?","<NUMBER>",x),words))
    return TaggedDocument(words= words,tags = [ article.pageid ])

with open('data/pages_with_id.data', 'rb') as file:
    pages: List[Page] = pickle.load(file)

pages = filter(lambda a: a.pageid.isnumeric(),pages)
with Pool(3) as p:
    cleaned = p.map(clean_text,pages)
documents =  list(cleaned)

model50  = Doc2Vec(documents,vector_size = 50, dm_concat = 1, window = 5,min_count = 100, workers = 3, epochs = 100)
model50.save('data/paragraphs50.data')
model100  = Doc2Vec(documents,vector_size = 100, dm_concat = 1, window = 5,min_count = 100, workers = 3, epochs = 100)
model100.save('data/paragraphs100.data')

model150  = Doc2Vec(documents,vector_size = 150, dm_concat = 1, window = 5,min_count = 100, workers = 3, epochs = 100)
model150.save('data/paragraphs150.data')
model200  = Doc2Vec(documents,vector_size = 200, dm_concat = 1, window = 5,min_count = 100, workers = 3, epochs = 100)
model200.save('data/paragraphs200.data')

import csv
import pickle
import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import wikipediaapi

Article = namedtuple('Article', ['number', 'title', 'pageid', 'namespace', 'length', 'touched'])
Page = namedtuple('Page', ['title', 'text'])
wiki = wikipediaapi.Wikipedia('en', timeout=60)
filename = 'history_depth3.csv'


def get_page(article: Article) -> Page:
    for x in range(10):
        try:
            wiki_page = wiki.page(article.title)
            text = wiki_page.text
            return Page(article.title, text)
        except Exception as e:
            print(x, repr(e))
            time.sleep(1)
    return Page(article.title, '')


@contextmanager
def timeit():
    start = time.time()
    yield
    print(time.time() - start, 's')


if __name__ == '__main__':
    with open(filename, 'r', encoding='utf-8') as file, ThreadPoolExecutor() as pool, timeit():
        articles = map(Article._make, csv.reader(file))
        pages = filter(lambda page: page.text, pool.map(get_page, articles))

    with open('pages', 'wb') as result_file:
        pickle.dump(list(pages), result_file)

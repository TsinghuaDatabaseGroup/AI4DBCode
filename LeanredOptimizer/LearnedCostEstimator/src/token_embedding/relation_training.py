import logging  # Setting up the loggings to monitor gensim
import multiprocessing
import pickle
from time import time  # To time our operations

from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt='%H:%M:%S', level=logging.INFO)
sentences = []

for path in ['aka_name.pkl', 'cast_info.pkl', 'company_name.pkl', 'info_type.pkl', 'link_type.pkl',
             'movie_info_idx.pkl', 'movie_keyword.pkl', 'name.pkl', 'role_type.pkl', 'aka_title.pkl', 'char_name.pkl',
             'company_type.pkl', 'keyword.pkl', 'movie_companies.pkl', 'movie_info.pkl', 'movie_link.pkl',
             'person_info.pkl', 'title.pkl']:
    print(path)
    with open(path, 'rb') as f:
        content = f.read()
        sentences += pickle.loads(content)
print('Token loading completely!')

cores = multiprocessing.cpu_count()
w2v_model = Word2Vec(min_count=5,
                     window=5,
                     size=500,
                     alpha=0.03,
                     min_alpha=0.0007,
                     negative=20,
                     workers=cores - 10)
t = time()
w2v_model.build_vocab(sentences, progress_per=10000)
print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

t = time()
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

w2v_model.save("word2vec.model")
print('model saved')

path = get_tmpfile("wordvectors.kv")
w2v_model.wv.save(path)
print('word saved')
'''
model.build_vocab(new_sentences, update=True)
model.train(new_sentences)
'''

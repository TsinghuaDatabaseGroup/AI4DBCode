import pickle
import re

from src.feature_extraction.database_loader import *

data = load_dataset('/home/sunji/cost_estimation/test_files_open_source/imdb_data_csv')

at_tokens = []
for idx, row in data["aka_title"].iterrows():
    if idx % 100 == 0:
        print('aka_title', idx, '/', len(data["aka_title"]))
    sentence = []
    sentence.append('at_id_' + str(row['id']))
    sentence.append('m_id_' + str(row['movie_id']))
    title = str(row['title'])
    if len(title) > 0:
        sentence.append('title_' + title)
    for t in title.split(' '):
        if len(t) > 0:
            sentence.append('title_' + t)
    sentence.append('k_id_' + str(row['kind_id']))
    publication_year = str(row['production_year'])
    if len(publication_year) > 0:
        sentence.append('year_' + publication_year)
    at_tokens.append(sentence)
print('aka_title generated')
with open('/home/sunji/learnedcardinality/string_words/aka_title.pkl', 'wb') as f:
    pickle.dump(at_tokens, f)
print('aka_title')

mk_tokens = []
for idx, row in data["movie_keyword"].iterrows():
    if idx % 100 == 0:
        print('movie_keyword', idx, '/', len(data["movie_keyword"]))
    sentence = []
    sentence.append('mk_id_' + str(row['id']))
    sentence.append('m_id_' + str(row['movie_id']))
    sentence.append('key_id_' + str(row['keyword_id']))
    mk_tokens.append(sentence)
print('movie_keyword generated')
with open('/home/sunji/learnedcardinality/string_words/movie_keyword.pkl', 'wb') as f:
    pickle.dump(mk_tokens, f)
print('movie_keyword')

ml_tokens = []
for idx, row in data["movie_link"].iterrows():
    if idx % 100 == 0:
        print('movie_link', idx, '/', len(data["movie_link"]))
    sentence = []
    sentence.append('ml_id_' + str(row['id']))
    sentence.append('m_id_' + str(row['movie_id']))
    sentence.append('lm_id_' + str(row['linked_movie_id']))
    sentence.append('lt_id_' + str(row['link_type_id']))
    ml_tokens.append(sentence)
print('movie_link generated')
with open('/home/sunji/learnedcardinality/string_words/movie_link.pkl', 'wb') as f:
    pickle.dump(ml_tokens, f)
print('movie_link')

chn_tokens = []
for idx, row in data['char_name'].iterrows():
    if idx % 100 == 0:
        print('char_name', idx, '/', len(data['char_name']))
    sentence = []
    sentence.append('chn_id_' + str(row['id']))
    name = str(row['name'])
    sentence.append('name_' + name)
    name = name.replace('-', ' ').replace(',', ' ')
    for token in name.split(' '):
        if len(token) > 0:
            sentence.append('name_' + token)
            if token[0].isupper():
                ll = len(token)
                for l in range(1, min(ll - 1, 4) + 1):
                    sentence.append('name_' + token[0:l])
    name_pcode_nf = str(row['name_pcode_nf'])
    surname_pcode = str(row['surname_pcode'])
    if len(name_pcode_nf) > 0 and name_pcode_nf[0].isupper():
        sentence.append('nf_' + name_pcode_nf[0])
    if len(surname_pcode) > 0 and surname_pcode[0].isupper():
        sentence.append('surname_' + surname_pcode[0])
    chn_tokens.append(sentence)
print('char_name generated')
with open('/home/sunji/learnedcardinality/string_words/char_name.pkl', 'wb') as f:
    pickle.dump(chn_tokens, f)
print('char_name')

mi_idx_tokens = []
for idx, row in data["movie_info_idx"].iterrows():
    if idx % 100 == 0:
        print('movie_info_idx', idx, '/', len(data["movie_info_idx"]))
    sentence = []
    sentence.append('mi_idx_id_' + str(row['id']))
    sentence.append('m_id_' + str(row['movie_id']))
    sentence.append('it_id_' + str(row['info_type_id']))
    info = str(row['info'])
    if len(info) > 0:
        sentence.append('info_' + info)
    mi_idx_tokens.append(sentence)
print('movie_info_idx generated')
with open('/home/sunji/learnedcardinality/string_words/movie_info_idx.pkl', 'wb') as f:
    pickle.dump(mi_idx_tokens, f)
print('movie_info_idx')

t_tokens = []
for idx, row in data["title"].iterrows():
    if idx % 100 == 0:
        print('title', idx, '/', len(data["title"]))
    sentence = []
    sentence.append('t_id_' + str(row['id']))
    title = str(row['title'])
    if len(title) > 0:
        sentence.append('title_' + title)
    for t in title.split(' '):
        if len(t) > 0:
            sentence.append('title_' + t)
    sentence.append('k_id_' + str(row['kind_id']))
    publication_year = str(row['production_year'])
    if len(publication_year) > 0:
        sentence.append('year_' + publication_year)
    t_tokens.append(sentence)
print('title generated')
with open('/home/sunji/learnedcardinality/string_words/title.pkl', 'wb') as f:
    pickle.dump(t_tokens, f)
print('title')

rt_tokens = []
for idx, row in data["role_type"].iterrows():
    sentence = []
    sentence.append('rt_id_' + str(row['id']))
    sentence.append('role_' + str(row['role']))
    rt_tokens.append(sentence)
print('role_type generated')
with open('/home/sunji/learnedcardinality/string_words/role_type.pkl', 'wb') as f:
    pickle.dump(rt_tokens, f)
print('role_type')

mc_tokens = []
for idx, row in data["movie_companies"].iterrows():
    if idx % 100 == 0:
        print('movie_companies', idx, '/', len(data["movie_companies"]))
    sentence = []
    sentence.append('mc_id_' + str(row['id']))
    sentence.append('m_id_' + str(row['movie_id']))
    sentence.append('c_id_' + str(row['company_id']))
    sentence.append('ct_id_' + str(row['company_type_id']))
    note = str(row['note'])
    for token in re.findall(r'\([^\)]*\)', note):
        sentence.append('note_' + token)
    if len(note) > 0:
        mc_tokens.append(sentence)
print('movie_companies generated')
with open('/home/sunji/learnedcardinality/string_words/movie_companies.pkl', 'wb') as f:
    pickle.dump(mc_tokens, f)
print('movie_companies')

it_tokens = []
print("info_type", len(data["info_type"]))
for idx, row in data["info_type"].iterrows():
    sentence = []
    sentence.append('it_id_' + str(row['id']))
    token = str(row['info'])
    sentence.append(token)
    if len(token) > 0:
        it_tokens.append(sentence)
print('info_type generated')
with open('/home/sunji/learnedcardinality/string_words/info_type.pkl', 'wb') as f:
    pickle.dump(it_tokens, f)
print('info_type')

ct_tokens = []
print("company_type", len(data["company_type"]))
for idx, row in data["company_type"].iterrows():
    sentence = []
    sentence.append('ct_id_' + str(row['id']))
    token = str(row['kind'])
    sentence.append(token)
    if len(token) > 0:
        ct_tokens.append(sentence)
print('company_type generated')
with open('/home/sunji/learnedcardinality/string_words/company_type.pkl', 'wb') as f:
    pickle.dump(ct_tokens, f)
print('company_type')

cn_tokens = []
print("company_name", len(data["company_name"]))
for idx, row in data["company_name"].iterrows():
    sentence = []
    sentence.append('cn_id_' + str(row['id']))
    name = str(row['name'])
    if len(name) > 0:
        sentence.append('cn_name_' + name)
    for token in name.split(' '):
        if len(token) > 0:
            sentence.append('cn_name_' + token)
    sentence.append('country_' + str(row['country_code']))
    if len(token) > 0:
        cn_tokens.append(sentence)
print('company_name generated')
with open('/home/sunji/learnedcardinality/string_words/company_name.pkl', 'wb') as f:
    pickle.dump(cn_tokens, f)
print('company_name')

k_tokens = []
print("keyword", len(data["keyword"]))
for idx, row in data["keyword"].iterrows():
    sentence = []
    sentence.append('key_id_' + str(row['id']))
    keyword = str(row['keyword'])
    sentence.append('keyword_' + keyword)
    for token in keyword.split('-'):
        sentence.append('keyword_' + token)
    if len(keyword) > 0:
        k_tokens.append(sentence)
print('keyword generated')
with open('/home/sunji/learnedcardinality/string_words/keyword.pkl', 'wb') as f:
    pickle.dump(k_tokens, f)
print('keyword')

mi_tokens = []
for idx, row in data["movie_info"].iterrows():
    if idx % 100 == 0:
        print('movie_info', idx, '/', len(data['movie_info']))
    sentence = []
    sentence.append('mi_id_' + str(row['id']))
    sentence.append('m_id_' + str(row['movie_id']))
    sentence.append('it_id_' + str(row['info_type_id']))
    sentence.append(str(row['info']))
    note = str(row['note'])
    for token in re.split(r"[\s\[\]\d'\(\),=#\?\.\-\{\}~+\*\\/]", note):
        sentence.append('note_' + token)
    mi_tokens.append(sentence)
print('movie_info generated')
with open('/home/sunji/learnedcardinality/string_words/movie_info.pkl', 'wb') as f:
    pickle.dump(mi_tokens, f)
print('movie_info')

n_tokens = []
for idx, row in data['name'].iterrows():
    if idx % 100 == 0:
        print('name', idx, '/', len(data['name']))
    sentence = []
    sentence.append('n_id_' + str(row['id']))
    sentence.append('gender_' + str(row['gender']))
    name = str(row['name'])
    name = name.replace('-', ' ').replace(',', ' ')
    for token in name.split(' '):
        if len(token) > 0:
            sentence.append('name_' + token)
            if token[0].isupper():
                ll = len(token)
                for l in range(1, min(ll - 1, 4) + 1):
                    sentence.append('name_' + token[0:l])
    name_pcode_cf = str(row['name_pcode_cf'])
    name_pcode_nf = str(row['name_pcode_nf'])
    surname_pcode = str(row['surname_pcode'])
    if len(name_pcode_cf) > 0 and name_pcode_cf[0].isupper():
        sentence.append('cf_' + name_pcode_cf[0])
    if len(name_pcode_nf) > 0 and name_pcode_nf[0].isupper():
        sentence.append('nf_' + name_pcode_nf[0])
    if len(surname_pcode) > 0 and surname_pcode[0].isupper():
        sentence.append('surname_' + surname_pcode[0])
    n_tokens.append(sentence)
print('name generated')
with open('/home/sunji/learnedcardinality/string_words/name.pkl', 'wb') as f:
    pickle.dump(n_tokens, f)
print('name')

an_tokens = []
for idx, row in data['aka_name'].iterrows():
    if idx % 100 == 0:
        print('aka_name', idx, '/', len(data['aka_name']))
    sentence = []
    sentence.append('an_id_' + str(row['id']))
    sentence.append('p_id_' + str(row['person_id']))
    name = str(row['name'])
    name = name.replace(',', ' ')
    tokens = []
    for token in name:
        if token.isalpha():
            tokens.append(token)
    tokens = list(set(tokens))
    for t in tokens:
        sentence.append('name_' + t)
    name_pcode_cf = str(row['name_pcode_cf'])
    name_pcode_nf = str(row['name_pcode_nf'])
    surname_pcode = str(row['surname_pcode'])
    if len(name_pcode_cf) > 0 and name_pcode_cf[0].isupper():
        sentence.append('cf_' + name_pcode_cf[0])
    if len(name_pcode_nf) > 0 and name_pcode_nf[0].isupper():
        sentence.append('nf_' + name_pcode_nf[0])
    if len(surname_pcode) > 0 and surname_pcode[0].isupper():
        sentence.append('surname_' + surname_pcode[0])
    an_tokens.append(sentence)
print('aka_name generated')
with open('/home/sunji/learnedcardinality/string_words/aka_name.pkl', 'wb') as f:
    pickle.dump(an_tokens, f)
print('aka_name')

lt_tokens = []
print("link_type", len(data["link_type"]))
for idx, row in data["link_type"].iterrows():
    sentence = []
    sentence.append('lt_id_' + str(row['id']))
    token = str(row['link'])
    sentence.append('link_' + token)
    if len(token) > 0:
        lt_tokens.append(sentence)
print('link_type generated')
with open('/home/sunji/learnedcardinality/string_words/link_type.pkl', 'wb') as f:
    pickle.dump(lt_tokens, f)
print('link_type')

pi_tokens = []
for idx, row in data["person_info"].iterrows():
    if idx % 100 == 0:
        print('person_info', idx, '/', len(data["person_info"]))
    sentence = []
    sentence.append('pi_id_' + str(row['id']))
    sentence.append('p_id_' + str(row['person_id']))
    sentence.append('it_id_' + str(row['info_type_id']))
    note = str(row['note'])
    if len(note) > 0:
        sentence.append('note_' + note)
    pi_tokens.append(sentence)
print('person_info generated')
with open('/home/sunji/learnedcardinality/string_words/person_info.pkl', 'wb') as f:
    pickle.dump(pi_tokens, f)
print('person_info')

ci_tokens = []
for idx, row in data["cast_info"].iterrows():
    if idx % 100 == 0:
        print('cast_info', idx, '/', len(data['cast_info']))
    sentence = []
    sentence.append('ci_id_' + str(row['id']))
    sentence.append('p_id_' + str(row['person_id']))
    sentence.append('m_id_' + str(row['movie_id']))
    sentence.append('pr_id_' + str(row['person_role_id']))
    note = str(row['note'])
    if len(note) > 0:
        sentence.append('note_' + token)
    for token in re.findall(r'\([^\)]*\)', note):
        sentence.append('note_' + token)
    sentence.append('r_id_' + str(row['role_id']))
    ci_tokens.append(sentence)
print('cast_info generated')
with open('/home/sunji/learnedcardinality/string_words/cast_info.pkl', 'wb') as f:
    pickle.dump(ci_tokens, f)
print('cast_info')

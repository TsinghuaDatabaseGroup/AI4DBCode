#!/bin/bash
set -ex

mkdir -p ../../train-test-data/imdbdata-num && pushd ../../train-test-data/imdbdata-num
wget -c http://homepages.cwi.nl/~boncz/job/imdb.tgz && tar -xvzf imdb.tgz && popd

python scripts/prepend_imdb_headers.py

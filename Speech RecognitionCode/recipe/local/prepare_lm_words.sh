#!/bin/bash
# Apache 2.0

set -e -o pipefail

. ./path.sh || die "path.sh expected";

local/train_lms_srilm.sh --train-text data/train/text data/ data/srilm_words

nl -nrz -w10  corpus/LM/train.txt | utils/shuffle_list.pl > data/local_words/external_text
local/train_lms_srilm.sh --train-text data/local_words/external_text data/ data/srilm_external_words

[ -d data/lang_test_words/ ] && rm -rf data/lang_test_words
cp -R data/lang_words data/lang_test_words
lm=data/srilm_words/lm.gz

local/arpa2G.sh $lm data/lang_test_words data/lang_test_words

exit 0;

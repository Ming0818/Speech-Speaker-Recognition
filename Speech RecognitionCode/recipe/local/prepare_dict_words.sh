#!/bin/bash
# Apache 2.0

corpus=$1
if [ -z "$corpus" ] ; then
    echo >&2 "The script $0 expects one parameter -- the location of the corpus"
    exit 1
fi
if [ ! -d "$corpus" ] ; then
    echo >&2 "The directory $corpus does not exist"
fi

mkdir -p data/lang_words data/local_words/dict

cp $corpus/lang_words/dict/lexicon.txt data/local_words/dict/lexicon.txt
cat data/local_words/dict/lexicon.txt | \
    perl -ane 'print join("\n", @F[1..$#F]) . "\n"; '  | \
    sort -u | grep -v 'SIL' > data/local_words/dict/nonsilence_phones.txt

touch data/local_words/dict/extra_questions.txt
touch data/local_words/dict/optional_silence.txt

echo "SIL"   > data/local_words/dict/optional_silence.txt
echo "SIL"   > data/local_words/dict/silence_phones.txt
echo "<UNK>" > data/local_words/dict/oov.txt

echo "Dictionary words preparation succeeded"

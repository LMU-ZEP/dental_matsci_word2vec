
# Dental & Materials Science Word2Vec Pipeline

This repository contains a reproducible pipeline for training Word2Vec word embeddings
on a corpus of dental and materials science PDF articles.

The pipeline:

1. **Extracts text** from PDF files.
2. **Cleans** common PDF artifacts (hyphenation, weird line breaks, invalid Unicode).
3. **Preprocesses** text with spaCy (tokenization, lemmatization, stopword & punctuation removal).
4. **Learns bigram phrases** with Gensim `Phrases` and applies them on-the-fly.
5. **Trains a Word2Vec model** (CBOW or Skip-Gram) on the bigrammed corpus and saves it.

The main script is assumed to be:

```bash
dental_word2vec_pipeline.py


Python dependencies are listed in requirements.txt:

* spacy – NLP pipeline for tokenization & lemmatization
* gensim – Phrases (bigrams) and Word2Vec
* nltk – English stopwords
* PyPDF2 – PDF text extraction
* ijson – streaming JSON parser (memory-efficient)

In addition, you must install:
* spaCy English model: en_core_web_sm
* NLTK stopwords data

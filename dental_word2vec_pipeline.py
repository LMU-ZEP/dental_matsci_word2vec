"""
Pipeline for building a Word2Vec model from dental and materials science PDFs.

Steps:
1. Extract text from PDFs and build a raw corpus (JSON list of strings).
2. Preprocess corpus: lowercase, tokenize, lemmatize, remove stopwords/punctuation.
3. Learn bigram phrases and apply them on-the-fly via BigramCorpus.
4. Train a Word2Vec model on the bigrammed corpus and save it.
"""

import json
import multiprocessing
import os
import random
import re
from typing import Iterable, List, Any

import ijson
import spacy
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from nltk.corpus import stopwords
from PyPDF2 import PdfReader


class BigramCorpus:
    """
    Lazy iterable over a JSON corpus that applies a bigram model on-the-fly.

    The underlying file is a JSON list of token lists:
        ["doc0 tokens ...", "doc1 tokens ...", ...]  (after preprocessing)

    Each iteration:
      * opens the file,
      * streams items via ijson,
      * applies bigram_model[doc],
      * yields the transformed list of tokens.

    This makes the corpus re-iterable and suitable for multiple epochs of training.
    """

    def __init__(self, file_path: str, bigram_model: Phraser):
        """
        Parameters
        ----------
        file_path : str
            Path to the JSON file containing a list of token lists.
        bigram_model : gensim.models.phrases.Phraser
            Trained bigram (or n-gram) model to apply to each document.
        """
        self.file_path = file_path
        self.bigram_model = bigram_model

    def __iter__(self) -> Iterable[List[str]]:
        with open(self.file_path, "r", encoding="utf-8") as f:
            for doc in ijson.items(f, "item"):
                # doc is expected to be a list of tokens
                yield self.bigram_model[doc]


# --- NLP setup ----------------------------------------------------------------

try:
    nlp = spacy.load("en_core_web_sm")
except OSError as exc:
    raise OSError(
        "SpaCy model 'en_core_web_sm' is not installed. "
        "Install it with: python -m spacy download en_core_web_sm"
    ) from exc

nlp.max_length = 3_000_000  # allow long documents

stop_words = set(stopwords.words("english"))


# --- Utility functions --------------------------------------------------------


def select_random_files(
    folder_path: str,
    percentage: float = 0.9,
    file_extensions: Iterable[str] | None = None,
) -> List[str]:
    """
    Select a random subset of files from a folder.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing files.
    percentage : float, optional
        Fraction of files to select (0.0–1.0). Default is 0.9.
    file_extensions : Iterable[str] or None, optional
        If given, only files with these extensions (case-insensitive) are considered.

    Returns
    -------
    List[str]
        List of selected file paths.
    """
    all_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]

    if file_extensions:
        ext_set = {ext.lower() for ext in file_extensions}
        all_files = [
            f for f in all_files if os.path.splitext(f)[1].lower() in ext_set
        ]

    if not all_files:
        return []

    sample_size = int(len(all_files) * percentage)
    if sample_size <= 0:
        return []

    selected_files = random.sample(all_files, sample_size)
    return selected_files


def reset_eof_of_pdf_return_stream(pdf_stream_in: List[bytes]) -> List[bytes]:
    """
    Given a raw PDF byte stream (list of lines), trim everything after the last %%EOF.

    This can be useful for repairing truncated/broken PDFs where extra bytes after EOF
    confuse parsers.

    Parameters
    ----------
    pdf_stream_in : list of bytes
        Lines read from a PDF file in binary mode.

    Returns
    -------
    list of bytes
        Truncated list up to and including the last line that contains '%%EOF'.
    """
    stream_len = len(pdf_stream_in)
    actual_line = stream_len
    for i, x in enumerate(pdf_stream_in[::-1]):
        if b"%%EOF" in x:
            actual_line = stream_len - i
            print(
                f"EOF found at line position {-i} = actual {actual_line}, "
                f"with value {x}"
            )
            break

    return pdf_stream_in[:actual_line]


def rewrite_pdf(filepath: str) -> str:
    """
    Attempt to repair a PDF by removing bytes after the last '%%EOF' marker.

    Parameters
    ----------
    filepath : str
        Path to the original PDF.

    Returns
    -------
    str
        Path to the repaired PDF (with '_fixed.pdf' suffix).
    """
    with open(filepath, "rb") as p:
        pdf_lines = p.readlines()

    trimmed_lines = reset_eof_of_pdf_return_stream(pdf_lines)
    new_filepath = filepath[:-4] + "_fixed.pdf"

    with open(new_filepath, "wb") as f:
        f.writelines(trimmed_lines)

    return new_filepath


def lemmatize_text(text: str) -> List[str]:
    """
    Lemmatize text using spaCy, remove English stopwords and punctuation.

    Parameters
    ----------
    text : str
        Input text (one document).

    Returns
    -------
    List[str]
        List of lemmatized tokens (lowercased, stopwords/punctuation removed).
    """
    doc = nlp(text)
    lemmas: List[str] = []

    for token in doc:
        # Skip punctuation and whitespace
        if token.is_punct or token.is_space:
            continue

        word = token.text.lower()
        if word in stop_words:
            continue

        lemma = token.lemma_.lower()
        lemmas.append(lemma)

    return lemmas


def fix_pdf_hyphenation(text: str) -> str:
    """
    Fix common hyphenation and line-break artifacts in PDF-extracted text.

    Operations:
    - Join words split with hyphens across line breaks, e.g., 'infor-\\nmation' -> 'information'.
    - Join words split by a plain line break, e.g., 'data\\nbase' -> 'data base'.
    - Normalize whitespace to single spaces.

    Parameters
    ----------
    text : str
        Raw text extracted from PDF pages.

    Returns
    -------
    str
        Cleaned text with fewer line-break artifacts.
    """
    # Fix hyphenated line breaks: infor-\nmation -> information
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

    # Join words split by a newline: data\nbase -> data base
    text = re.sub(r"(\w+)\s*\n\s*(\w+)", r"\1 \2", text)

    # Normalize runs of whitespace to a single space
    text = re.sub(r"\s+", " ", text)

    return text


def remove_surrogates(s: str) -> str:
    """
    Remove Unicode surrogate code points that may appear in PDF-derived text.

    Surrogates are in the range U+D800 to U+DFFF; they are invalid as standalone
    codepoints in UTF-8/UTF-32 and may cause encoding errors in IO or JSON.

    Parameters
    ----------
    s : str
        Input string.

    Returns
    -------
    str
        String with surrogate code points removed.
    """
    return re.sub(r"[\ud800-\udfff]", "", s)


def clean(obj: Any) -> Any:
    """
    Recursively remove Unicode surrogate code points from strings
    inside nested structures (lists, dicts).

    Parameters
    ----------
    obj : Any
        Arbitrary object: str, list, dict, or other.

    Returns
    -------
    Any
        Same structure, but with surrogates removed from any contained strings.
    """
    if isinstance(obj, str):
        return remove_surrogates(obj)
    if isinstance(obj, list):
        return [clean(item) for item in obj]
    if isinstance(obj, dict):
        return {clean(k): clean(v) for k, v in obj.items()}
    return obj


# --- Corpus building ----------------------------------------------------------


def get_corpus(
    data_folders: List[str],
    save_name: str,
    percentage: float = 1.0,
    failed_log_path: str = "failed_process_pdf.log",
) -> None:
    """
    Build a raw text corpus from a set of folders with PDF files.

    For each PDF:
    - try to read all pages with PyPDF2,
    - concatenate page texts,
    - fix hyphenation / line breaks,
    - append resulting document string to a list.

    Finally, the corpus is saved as a JSON list of strings.

    Parameters
    ----------
    data_folders : list of str
        List of folder paths containing PDF files.
    save_name : str
        Path to save the resulting corpus JSON.
    percentage : float, optional
        Fraction of files within each folder to process (0.0–1.0). Default is 1.0.
    failed_log_path : str, optional
        Path to a log file where failures to read PDFs are recorded.
    """
    print("Creating a corpus")
    corpus: List[str] = []
    processed_docs = 0

    for data_folder in data_folders:
        if percentage < 1.0:
            selected_files = select_random_files(
                folder_path=data_folder, percentage=percentage
            )
        else:
            selected_files = [
                os.path.join(data_folder, f)
                for f in os.listdir(data_folder)
                if os.path.isfile(os.path.join(data_folder, f))
            ]

        for file_path in selected_files:
            try:
                reader = PdfReader(file_path)
            except Exception as e:  # noqa: BLE001 – log all PDF read errors
                with open(failed_log_path, "a", encoding="utf-8") as log_f:
                    print(e, file=log_f)
                    print(file_path, file=log_f)
                continue

            texts: List[str] = []
            try:
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        texts.append(page_text)
            except Exception as e:  # noqa: BLE001 – log all text extraction errors
                with open(failed_log_path, "a", encoding="utf-8") as log_f:
                    print(e, file=log_f)
                    print(file_path, file=log_f)
                continue

            if not texts:
                continue

            text = "\n".join(texts)
            cleaned_text = fix_pdf_hyphenation(text)
            processed_docs += 1
            corpus.append(cleaned_text)

    print("Processed docs count:", processed_docs)

    with open(save_name, "w", encoding="utf-8") as f:
        json.dump(clean(corpus), f, ensure_ascii=False)

    print(f"Corpus saved to: {save_name}")


def preprocess_corpus_line_by_line(
    corpus_path: str,
    preprocess_corpus_path: str,
) -> None:
    """
    Preprocess a JSON corpus line-by-line and save the tokenized corpus.

    Input:
      - JSON file with a list of raw document strings.

    Output:
      - JSON file with a list of lists of tokens (lemmatized, lowercased,
        stopwords/punctuation removed).

    Streaming is done via ijson to avoid loading everything into memory.

    Parameters
    ----------
    corpus_path : str
        Path to input JSON corpus (list of strings).
    preprocess_corpus_path : str
        Path to output JSON file with token lists.
    """
    print("Preprocessing corpus")
    print("---------------------------")

    with open(corpus_path, "r", encoding="utf-8") as f_in, open(
        preprocess_corpus_path, "w", encoding="utf-8"
    ) as f_out:
        docs = ijson.items(f_in, "item")
        f_out.write("[\n")  # start of JSON list

        first = True
        for i, doc in enumerate(docs):
            if i % 10_000 == 0:
                print(f"Preprocessed {i} documents")

            # doc is a raw string from the original corpus
            tokens = lemmatize_text(doc)

            if not first:
                f_out.write(",\n")
            json.dump(tokens, f_out, ensure_ascii=False)
            first = False

        f_out.write("\n]")  # end of JSON list

    print(f"Preprocessed corpus saved to: {preprocess_corpus_path}")


def stream_json_list_of_lists(file_path: str) -> Iterable[List[str]]:
    """
    Stream a JSON file containing a list of lists (e.g., tokenized documents).

    Uses ijson to avoid loading the full list into memory.

    Parameters
    ----------
    file_path : str
        Path to the JSON file with a list of lists.

    Yields
    ------
    list of str
        Next document (list of tokens).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        for item in ijson.items(f, "item"):
            yield item


# --- Word2Vec training --------------------------------------------------------


def train_word2vec(
    corpus_with_bigrams: Iterable[List[str]],
    skip_gram: bool = False,
    n_epochs: int = 10,
    save_path: str = "w2v_model.model",
    min_count: int = 15,
    window: int = 8,
    vector_size: int = 100,
    sample: float = 6e-5,
    alpha: float = 0.03,
    min_alpha: float = 0.0007,
    negative: int = 20,
    seed: int = 42,
) -> None:
    """
    Train a Word2Vec model on a bigrammed corpus and save it.

    Parameters
    ----------
    corpus_with_bigrams : Iterable[list[str]]
        Re-iterable corpus of token lists with bigrams already applied.
        (e.g., an instance of BigramCorpus)
    skip_gram : bool, optional
        If True, use Skip-Gram (sg=1); if False, use CBOW (sg=0). Default is False.
    n_epochs : int, optional
        Number of training epochs. Default is 10.
    save_path : str, optional
        Path to save the trained Word2Vec model. Default is 'w2v_model.model'.
    min_count : int, optional
        Word2Vec min_count parameter. Default is 15.
    window : int, optional
        Word2Vec window size. Default is 8.
    vector_size : int, optional
        Dimensionality of the word vectors. Default is 100.
    sample : float, optional
        Subsampling rate for frequent words. Default is 6e-5.
    alpha : float, optional
        Initial learning rate. Default is 0.03.
    min_alpha : float, optional
        Minimum learning rate. Default is 0.0007.
    negative : int, optional
        Number of negative samples. Default is 20.
    seed : int, optional
        Random seed for reproducibility. Default is 42.
    """
    cores = multiprocessing.cpu_count()

    w2v_model = Word2Vec(
        min_count=min_count,
        window=window,
        vector_size=vector_size,
        sample=sample,
        alpha=alpha,
        min_alpha=min_alpha,
        negative=negative,
        sg=1 if skip_gram else 0,
        workers=max(1, cores - 1),
        seed=seed,
    )

    print("Building Word2Vec vocabulary...")
    w2v_model.build_vocab(corpus_with_bigrams, progress_per=10_000)
    print(f"Vocabulary size: {len(w2v_model.wv)}")
    print(f"Number of documents (corpus_count): {w2v_model.corpus_count}")

    print("Training Word2Vec model...")
    w2v_model.train(
        corpus_with_bigrams,
        total_examples=w2v_model.corpus_count,
        epochs=n_epochs,
        report_delay=10,
    )

    w2v_model.save(save_path)
    print(f"Word2Vec model saved to: {save_path}")


# --- Main script --------------------------------------------------------------


def main() -> None:
    """
    Main entry point: builds corpus, preprocesses it, learns bigrams,
    and trains a Word2Vec model on the resulting bigram corpus.
    """
    # Paths / configuration
    w2v_save_path = "test.model"

    # List of paths to folders with dental and materials science articles (PDF)
    data_folders = [
       "path_to_folder_with_pdf_articles"
    ]

    # Path to save the raw corpus (list of raw document strings)
    corpus_save_path = "test.json"

    # Path to save the preprocessed corpus (list of token lists)
    prep_corpus_save_path = "prep_test.json"

    # 1. Build raw text corpus from PDFs
    get_corpus(data_folders, corpus_save_path, percentage=1.0)

    # 2. Preprocess corpus: lemmatize, remove stopwords/punctuation
    preprocess_corpus_line_by_line(corpus_save_path, prep_corpus_save_path)

    # 3. Learn bigrams on the tokenized corpus
    corpus_stream = stream_json_list_of_lists(prep_corpus_save_path)
    phrases = Phrases(corpus_stream, min_count=30, progress_per=10_000)
    bigram = Phraser(phrases)

    # 4. Create a lazy bigrammed corpus wrapper
    bigram_corpus = BigramCorpus(prep_corpus_save_path, bigram)

    # 5. Train Word2Vec model on bigram corpus (CBOW by default)
    train_word2vec(
        bigram_corpus,
        skip_gram=False,
        n_epochs=30,
        save_path=w2v_save_path,
    )


if __name__ == "__main__":
    main()

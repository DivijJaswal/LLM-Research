import torch
from transformers import pipeline
from huggingface_hub import login
from transformers import AutoTokenizer
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import spacy
from sklearn.decomposition import TruncatedSVD

def shorten_text(text, chunk_size, method, summarizer = None):
    """
    Shortens the text using the specified method.
    :param text: The original text to shorten.
    :param chunk_size: The size of each chunk if needed.
    :param method: Method to shorten the text.
    :return: Shortened text.
    """
    if method == "clipping":
        return text[:chunk_size]

    elif method == "iterative":
        chunks = [' '.join(text.split()[i:i+chunk_size]) for i in range(0, len(text.split()), chunk_size)]
        summarized_chunks = [summarizer(chunk, min_length=50, max_length=100, do_sample=False)[0]['summary_text'] for chunk in chunks]
        return " ".join(summarized_chunks)

    elif method == "random_removal":
        words = text.split()
        while len(words) > chunk_size:
            index_to_remove = random.randint(0, len(words) - 1)
            del words[index_to_remove]
        return " ".join(words)

    elif method == "sentence_ranking":
        return tfidf_sentence_ranking(text, chunk_size)

    elif method == "sliding_window":
        return sliding_window(text, chunk_size)

    elif method == "entity_filtering":
        return entity_filtering(text, chunk_size)

    elif method == "summarize_summary":
        return summarize_summary(text, chunk_size)

    elif method == "lsa":
        return lsa_text_summarization(text, chunk_size)

def tfidf_sentence_ranking(text, chunk_size):
    sentences = text.split('. ')
    vectorizer = TfidfVectorizer().fit_transform(sentences)
    vectors = vectorizer.toarray()
    sentence_scores = np.sum(vectors, axis=1)

    ranked_sentences = sorted(((score, i, s) for i, (score, s) in enumerate(zip(sentence_scores, sentences))), reverse=True)
    selected_sentences = [s for _, _, s in ranked_sentences[:int(chunk_size / 20)]]
    return '. '.join(selected_sentences)

def sliding_window(text, chunk_size, overlap=100, summarizer = None):
    words = text.split()
    windows = []
    for i in range(0, len(words), chunk_size - overlap):
        window = words[i:i+chunk_size]
        windows.append(" ".join(window))

    summarized_windows = [summarizer(w, min_length=50, max_length=100, do_sample=False)[0]['summary_text'] for w in windows]
    return " ".join(summarized_windows)

def entity_filtering(text, chunk_size):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    important_sentences = []
    for sent in doc.sents:
        if any(ent.label_ in ["PERSON", "ORG", "GPE", "DATE"] for ent in sent.ents):
            important_sentences.append(sent.text)
        if len(important_sentences) >= chunk_size / 20:
            break
    return " ".join(important_sentences)


def summarize_summary(text, chunk_size, min_length=200, summarizer = None):
    """
    Recursively summarize the text until it is under a desired length.
    """
    summarized_text = text
    iteration = 0
    while len(summarized_text) > chunk_size:
        print(f"Iteration {iteration + 1}: Text too long. Summarizing again.")
        summarized_text = summarizer(summarized_text, num_beams=5, min_length=min_length, max_length=chunk_size, do_sample=False)[0]['summary_text']
        iteration += 1
    return summarized_text

def lsa_text_summarization(text, chunk_size):
    """
    Use Latent Semantic Analysis (LSA) to extract the most important concepts from the text and return the most relevant sentences.
    """
    sentences = text.split('. ')

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)

    lsa_model = TruncatedSVD(n_components=1, n_iter=100)
    lsa_model.fit(X)
    lsa_scores = lsa_model.transform(X)

    # Rank sentences by their relevance to the main topics
    ranked_sentences = sorted(((lsa_scores[i, 0], s) for i, s in enumerate(sentences)), reverse=True)

    # Select the top N sentences based on LSA scores
    selected_sentences = [s for _, s in ranked_sentences[:int(chunk_size / 20)]]
    return '. '.join(selected_sentences)

import torch
from transformers import pipeline
from transformers import AutoTokenizer
from huggingface_hub import login
import evaluate
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import spacy
from sklearn.decomposition import TruncatedSVD
from transformers import logging
from contextlib import redirect_stdout
import os

import nltk
nltk.download('wordnet', quiet=True)
logging.set_verbosity_error()
login(token = "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", add_to_git_credential=True)
tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")

summarizer = pipeline("summarization",tokenizer=tokenizer, model="google/pegasus-xsum")

references = []

line_breaker = ".......line breaker........."
mn = 0
mx = 0
def shorten_text(text, chunk_size, method):
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
        summarized_chunks = [summarizer(chunk, min_length=mn, max_length=mx, do_sample=False)[0]['summary_text'] for chunk in chunks]
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

def sliding_window(text, chunk_size, overlap=100):
    words = text.split()
    windows = []
    for i in range(0, len(words), chunk_size - overlap):
        window = words[i:i+chunk_size]
        windows.append(" ".join(window))

    summarized_windows = [summarizer(w, min_length=mn, max_length=mx, do_sample=False)[0]['summary_text'] for w in windows]
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


def summarize_summary(text, chunk_size):
    """
    Recursively summarize the text until it is under a desired length.
    """
    summarized_text = text
    iteration = 0
    while len(summarized_text) > chunk_size:
        print(f"Iteration {iteration + 1}: Text too long. Summarizing again.")
        last_text=summarized_text
        summarized_text = shorten_text(last_text,512,"iterative")
        if((last_text==summarized_text) or (len(last_text) < len(summarized_text))):
            break
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

def scores(predictions):
    print("bleu")
    bleu = evaluate.load('bleu')
    results = bleu.compute(predictions=predictions, references=references)
    print(results)

    print("rouge")
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=predictions,references=references)
    print(results)
 
    print("bertscore")
    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(predictions=predictions, references=references, lang="en")
    print(results)

    print("meteor")
    meteor = evaluate.load('meteor')
    results = meteor.compute(predictions=predictions, references=references)
    print(results)
    
def generate_summaries(text, num_beams):
    print("clipping")
    clipped_text = shorten_text(text, 512, "clipping")
    short_summary = summarizer(clipped_text ,num_beams, min_length = mn, max_length =mx,do_sample=False)
    print(short_summary)
    predictions = [item['summary_text'] for item in short_summary]
    scores(predictions=predictions)

    print("iterative")
    short_summary = shorten_text(text,512,"iterative")
    print(short_summary)
    predictions = [short_summary]
    scores(predictions=predictions)

    print("Random Removal")
    removed_text = shorten_text(text,512,"random_removal")
    short_summary = summarizer(removed_text ,num_beams, min_length = mn, max_length =mx,do_sample=False)
    print(short_summary) 
    predictions = [item['summary_text'] for item in short_summary]
    scores(predictions=predictions)

    # sentence ranking not working
    # print("Sentence Ranking")
    # ranked_text = shorten_text(text,512,"sentence_ranking")
    # short_summary = summarizer(ranked_text ,num_beams, min_length = 50, max_length =100,do_sample=False)
    # print(short_summary)
    # predictions = [item['summary_text'] for item in short_summary]
    # scores(predictions=predictions)
    # print(line_breaker)

    print("Sliding Window")
    short_summary = shorten_text(text,512,"sliding_window")
    print(short_summary)
    predictions = [short_summary]
    scores(predictions=predictions)

    print("Entity Filtering")
    filtered_text = shorten_text(text,512,"entity_filtering")
    short_summary = summarizer(filtered_text ,num_beams, min_length = mn, max_length =mx,do_sample=False)
    print(short_summary)
    predictions = [item['summary_text'] for item in short_summary]
    scores(predictions=predictions)

    #lsa  not working
    # print("LSA")
    # lsa_text = shorten_text(text,512,"lsa")
    # short_summary = summarizer(lsa_text ,num_beams, min_length = 50, max_length =100,do_sample=False)
    # print(short_summary)
    # predictions = [item['summary_text'] for item in short_summary]
    # scores(predictions=predictions)
    # print(line_breaker)

    print("summarize_summary")
    short_summary = shorten_text(text,512,"summarize_summary")
    print(short_summary)
    predictions = [short_summary]
    scores(predictions=predictions)
        

def summarize_text(input_file,num_beams):
    
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()
    global mn, mx, references
    print("google/pegasus-xsum")    
    print("\n Short Summaries \n")
    mn = 50
    mx = 100
    references = ['''
Strategy for human spaceflight in Low Earth orbit

To develop a national strategy for human spaceflight in Low Earth Orbit (LEO), there are two key goals, returning Americans to the Moon and enabling commercial spaceflight. The strategy emphasizes international partnerships and a sustainable approach to space activities that doesn’t rely only on federal funding. The strategies align with directives that promote U.S. leadership in space exploration and aim to encourage economic growth, space research, and human expansion in space.

    ''']
    generate_summaries(text, num_beams)
    
    print("\n Medium Summaries \n")
    mn = 100
    mx = 150
    references = ['''
   Strategy for human spaceflight in Low Earth orbit

To develop a national strategy for human spaceflight in Low Earth Orbit (LEO), there are two key goals, returning Americans to the Moon and enabling commercial spaceflight.
The U.S. aims to collaborate internationally and promote sustainable commercial human spaceflight without federal dependency. In 2018, the National Space Council tasked NASA, the Department of State, and Commerce with developing a strategy for LEO human spaceflight, aligning with Space Policy Directives. The future human spaceflight strategy in LEO and economic growth aligns with directives, shaped through interagency collaboration and coordination with the Executive Office of the President. We have submitted our strategy to the National Space Council, essential for U.S. economic growth, global leadership, and humanity's future in space exploration.

    ''']
    generate_summaries(text, num_beams)
    
    print("\n Detailed Summaries \n")
    mn = 150
    mx = 200
    references = ['''
  Strategy for human spaceflight in Low Earth orbit

To develop a national strategy for human spaceflight in Low Earth Orbit (LEO), there are two key goals, returning Americans to the Moon and enabling commercial spaceflight.
The U.S. aims to collaborate internationally and promote sustainable commercial human spaceflight without federal dependency. In 2018, the National Space Council tasked NASA, the Department of State, and Commerce with developing a strategy for LEO human spaceflight, aligning with Space Policy Directives. The future human spaceflight strategy in LEO and economic growth aligns with directives, shaped through interagency collaboration and coordination with the Executive Office of the President, the goal for human spaceflight in LEO are as follow:
·  	Maintaining a continuous U.S. presence in LEO.
·  	Creating a supportive regulatory environment for commercial space ventures. Advancing spaceflight technologies for long-term missions.
·  	Expanding commercial opportunities through international partnerships.
We have submitted our strategy to the National Space Council, essential for U.S. economic growth, global leadership, and humanity's future in space exploration.

    ''']
    generate_summaries(text, num_beams)

    
def main():
    input_file="./text1.txt"
    print(input_file)
    num_beams=5
    summarize_text(input_file,num_beams)

if __name__ == '__main__':
    main()

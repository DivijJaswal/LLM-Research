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
file = open("./output.txt", "a")
references = []
mn = 0
mx = 0

def shorten_text(text, chunk_size, method, summarizer):
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
        return sliding_window(text, chunk_size, summarizer)

    elif method == "entity_filtering":
        return entity_filtering(text, chunk_size)

    elif method == "summarize_summary":
        return summarize_summary(text, chunk_size, summarizer)

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

def sliding_window(text, chunk_size, summarizer, overlap=100):
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


def summarize_summary(text, chunk_size, summarizer):
    """
    Recursively summarize the text until it is under a desired length.
    """
    summarized_text = text
    iteration = 0
    while len(summarized_text) > chunk_size:
        file.write("\nIteration {iteration + 1}: Text too long. Summarizing again.\n")
        last_text=summarized_text
        summarized_text = shorten_text(last_text,512,"iterative", summarizer)
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
    file.write("\nbleu\n")
    bleu = evaluate.load('bleu')
    results = bleu.compute(predictions=predictions, references=references)
    file.write(str(results))

    file.write("\nrouge\n")
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=predictions,references=references)
    file.write(str(results))
 
    file.write("\nbertscore\n")
    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(predictions=predictions, references=references, lang="en")
    file.write(str(results))

    file.write("\nmeteor\n")
    meteor = evaluate.load('meteor')
    results = meteor.compute(predictions=predictions, references=references)
    file.write(str(results))
    
def generate_summaries(text, num_beams, summarizer):
    file.write("\n\nclipping\n\n")
    clipped_text = shorten_text(text, 512, "clipping", summarizer)
    short_summary = summarizer(clipped_text ,num_beams, min_length = mn, max_length =mx,do_sample=False)
    file.write(str(short_summary))
    predictions = [item['summary_text'] for item in short_summary]
    scores(predictions=predictions)

    file.write("\n\niterative\n\n")
    short_summary = shorten_text(text,512,"iterative", summarizer)
    file.write(str(short_summary))
    predictions = [short_summary]
    scores(predictions=predictions)

    file.write("\n\nRandom Removal\n\n")
    removed_text = shorten_text(text,512,"random_removal", summarizer)
    short_summary = summarizer(removed_text ,num_beams, min_length = mn, max_length =mx,do_sample=False)
    file.write(str(short_summary)) 
    predictions = [item['summary_text'] for item in short_summary]
    scores(predictions=predictions)

    # sentence ranking not working
    # file.write("\n\nSentence Ranking\n\n")
    # ranked_text = shorten_text(text,512,"sentence_ranking", summarizer)
    # short_summary = summarizer(ranked_text ,num_beams, min_length = 50, max_length =100,do_sample=False)
    # file.write(str(short_summary))
    # predictions = [item['summary_text'] for item in short_summary]
    # scores(predictions=predictions)

    file.write("\n\nSliding Window\n\n")
    short_summary = shorten_text(text,512,"sliding_window", summarizer)
    file.write(str(short_summary))
    predictions = [short_summary]
    scores(predictions=predictions)

    file.write("\n\nEntity Filtering\n\n")
    filtered_text = shorten_text(text,512,"entity_filtering", summarizer)
    short_summary = summarizer(filtered_text ,num_beams, min_length = mn, max_length =mx,do_sample=False)
    file.write(str(short_summary))
    predictions = [item['summary_text'] for item in short_summary]
    scores(predictions=predictions)

    #lsa  not working
    # file.write("\n\nLSA\n\n")
    # lsa_text = shorten_text(text,512,"lsa", summarizer)
    # short_summary = summarizer(lsa_text ,num_beams, min_length = 50, max_length =100,do_sample=False)
    # file.write(str(short_summary))
    # predictions = [item['summary_text'] for item in short_summary]
    # scores(predictions=predictions)

    file.write("\n\nsummarize_summary\n\n")
    short_summary = shorten_text(text,512,"summarize_summary", summarizer)
    file.write(str(short_summary))
    predictions = [short_summary]
    scores(predictions=predictions)
        
def generate(text, num_beams, summarizer):
    global mn, mx, references
    file.write("\n Short Summaries \n")
    mn = 50
    mx = 100
    references = ['''
Strategy for human spaceflight in Low Earth orbit
 
The sustainable foundations for a new era of commercial spaceflight in low Earth orbit (LEO) are being developed as the linchpins of our national human spaceflight policy. The need for such a national strategy for human spaceflight in LEO and economic growth in space was recognized by the National Space Council. Human spaceflight goals have been defined to ensure presence, regulations, research, and commercial opportunities expanded. The strategy essential for U.S. economic growth, global leadership, and humanity's future in space exploration, has been submitted by us to the National Space Council. 
    ''']
    generate_summaries(text, num_beams, summarizer)
    
    file.write("\n Medium Summaries \n")
    mn = 100
    mx = 150
    references = ['''
  Strategy for human spaceflight in Low Earth orbit
 
The sustainable foundations for a new era of commercial spaceflight in low Earth orbit (LEO) are being developed as the linchpins of our national human spaceflight policy.
 
The need for such a national strategy for human spaceflight in LEO and economic growth in space was recognized by the National Space Council.
 
The human spaceflight goals has been defined as:
A continuous U.S. presence in LEO will be achieved, involving government astronauts and private citizens.
A regulatory environment in LEO that enables commercial activities will be created.
Human spaceflight research in LEO will be conducted to advance the technology and systems required for long-duration spaceflight,
Commercial opportunities will be expanded through international partnerships and engagement.
 
The strategy essential for U.S. economic growth, global leadership, and humanity's future in space exploration, has been submitted by us to National Space Council,

    ''']
    generate_summaries(text, num_beams, summarizer)
    
    file.write("\n Detailed Summaries \n")
    mn = 150
    mx = 200
    references = ['''
  Strategy for human spaceflight in Low Earth orbit
 
The sustainable foundations for a new era of commercial spaceflight in low Earth orbit (LEO) are being developed as the linchpins of our national human spaceflight policy.
 
The need for such a national strategy for human spaceflight in LEO and economic growth in space was recognized by the National Space Council. The Department of State, and Commerce, and NASA were tasked by National Space Council with developing a strategy for LEO human spaceflight,
 
Our strategy for the future of human spaceflight in LEO and economic growth in space will operate within the context of these directives.
 
The human spaceflight goals has been defined as:
A continuous U.S. presence in LEO will be achieved, involving government astronauts and private citizens.
A regulatory environment in LEO that enables commercial activities will be created.
Human spaceflight research in LEO will be conducted to advance the technology and systems required for long-duration spaceflight,
Commercial opportunities will be expanded through international partnerships and engagement.
 
The strategy essential for U.S. economic growth, global leadership, and humanity's future in space exploration, has been submitted by us to National Space Council,

    ''']
    generate_summaries(text, num_beams, summarizer)
    
def summarize_text(input_file,num_beams):
    
    with open(input_file, 'r', encoding='utf-8') as fil:
        text = fil.read()
    
    file.write("\n\nallenai/led-base-16384\n\n")    
    tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
    summarizer = pipeline("summarization",tokenizer=tokenizer, model="allenai/led-base-16384")
    generate(text, num_beams, summarizer)
    
    file.write("\n\nfacebook/bart-large-cnn\n\n")    
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    summarizer = pipeline("summarization",tokenizer=tokenizer, model="facebook/bart-large-cnn")
    generate(text, num_beams, summarizer)
    
    file.write("\n\ngoogle/flan-t5-base\n\n")    
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    summarizer = pipeline("summarization",tokenizer=tokenizer, model="google/flan-t5-base")
    generate(text, num_beams, summarizer)
    
    file.write("\n\ngoogle/pegasus-cnn_dailymail\n\n")    
    tokenizer = AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
    summarizer = pipeline("summarization",tokenizer=tokenizer, model="google/pegasus-cnn_dailymail")
    generate(text, num_beams, summarizer)
    
    file.write("\n\ngoogle/pegasus-xsum\n\n")    
    tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
    summarizer = pipeline("summarization",tokenizer=tokenizer, model="google/pegasus-xsum")
    generate(text, num_beams, summarizer)
    
    file.write("\n\ngoogle-t5/t5-base\n\n")    
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
    summarizer = pipeline("summarization",tokenizer=tokenizer, model="google-t5/t5-base")
    generate(text, num_beams, summarizer)

    
def main():
    input_file="./input.txt"
    file.write(input_file)
    num_beams=5
    summarize_text(input_file,num_beams)
    
if __name__ == '__main__':
    main()

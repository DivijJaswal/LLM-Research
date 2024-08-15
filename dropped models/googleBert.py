import torch
from transformers import pipeline
from huggingface_hub import login
from transformers import AutoTokenizer
# can't use because max token length for this is 512

login(token = "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")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
summarizer = pipeline("summarization",tokenizer=tokenizer, model="google-bert/bert-base-uncased")

def chunk_text(text, max_length=512):
    tokens = tokenizer.encode(text, truncation=False)
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    return chunks

def summarize_chunks(chunks,num_beams):
    summaries = []
    for chunk in chunks:
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        summary = summarizer(chunk_text,num_beams, max_length=130, min_length=30, do_sample=False,max_new_tokens=1000)[0]['summary_text']
        summaries.append(summary)
    return " ".join(summaries)

def summarize_text(text,num_beams):
    chunks = chunk_text(text)
    summary = summarize_chunks(chunks,num_beams)
    return summary


def summarize_text1(input_file,num_beams):
    
    
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()
    # summary = summarizer(text ,num_beams,do_sample=False,max_new_tokens=1000)    
    # summary = summarizer(text ,num_beams, max_length =110 , min_length =90,do_sample=False,max_new_tokens=1000)
    # summary = summarizer(text ,num_beams, max_length =220 , min_length =200,do_sample=False)
    # summary = summarizer(text ,num_beams, max_length =330 , min_length =310,do_sample=False)
    summary = summarize_text(text,num_beams)
    print(summary)
    
def main():
    input_file="./text1.txt"
    print(input_file)
    num_beams=5
    summarize_text1(input_file,num_beams)

if __name__ == '__main__':
    main()
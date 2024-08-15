import torch
from transformers import pipeline
from transformers import AutoTokenizer
from huggingface_hub import login
import evaluate

login(token = "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")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
summarizer = pipeline("summarization",tokenizer=tokenizer, model="google-bert/bert-base-uncased")

def chunk_text(text, max_length=512):
    words = text.split()
    current_chunk = ""
    chunks = []
    for word in words:
        current_chunk+=word+" "
        if(len(current_chunk)>2*max_length):
            tokens = tokenizer.encode(current_chunk, truncation=False)
            chunks.append(tokens)
            current_chunk=""
    # print(chunks)
    return chunks
    #     if len(tokenizer.encode(" ".join(current_chunk), add_special_tokens=False)) >= max_length:
    #         chunks.append(" ".join(current_chunk[:-1]))
    #         current_chunk = [word]
        
    # if current_chunk:
    #     chunks.append(" ".join(current_chunk))


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
    # print(summary)


# def summarize_text(input_file,num_beams):
#     login(token = "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")
#     tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

#     summarizer = pipeline("summarization",tokenizer=tokenizer, model="facebook/bart-large-cnn")
#     with open(input_file, 'r', encoding='utf-8') as file:
#         text = file.read()
#     short_summary = summarizer(text ,num_beams, min_length = 50, max_length =100,do_sample=False)
#     medium_summary = summarizer(text ,num_beams, min_length = 100, max_length =150,do_sample=False)
#     large_summary = summarizer(text ,num_beams, min_length = 150, max_length =200,do_sample=False)
#     # predictions = [item['summary_text'] for item in summary]
#     # predictions = summary.summary_text
#     # bleu = evaluate.load('bleu')
#     # results = bleu.compute(predictions=predictions, references=references)

#     print(short_summary)
#     print(medium_summary)
#     print(large_summary)
#     # print(results)

    
def main():
    input_file="./text1.txt"
    print(input_file)
    num_beams=5
    summarize_text1(input_file,num_beams)

if __name__ == '__main__':
    main()

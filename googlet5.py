import torch
from transformers import pipeline
from huggingface_hub import login
from transformers import AutoTokenizer

def summarize_text(input_file,num_beams):
    
    login(token = "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")

    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
    summarizer = pipeline("summarization", tokenizer=tokenizer,model="google-t5/t5-base")
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()
    # summary = summarizer(text ,num_beams,do_sample=False)    
    short_summary = summarizer(text ,num_beams, min_length = 50, max_length =100,do_sample=False)
    medium_summary = summarizer(text ,num_beams, min_length = 100, max_length =150,do_sample=False)
    large_summary = summarizer(text ,num_beams, min_length = 150, max_length =200,do_sample=False)

    print(short_summary)
    print(medium_summary)
    print(large_summary)

    # print(summary)
def main():
    input_file="./text1.txt"
    print(input_file)
    num_beams=5
    summarize_text(input_file,num_beams)

if __name__ == '__main__':
    main()
import torch
from transformers import pipeline
from huggingface_hub import login
from transformers import AutoTokenizer
# 11.31 GB  no tokenizer
def summarize_text(input_file,num_beams):
    
    login(token = "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")

    tokenizer = AutoTokenizer.from_pretrained("apple/OpenELM-3B-Instruct", trust_remote_code=True)
    summarizer = pipeline("summarization", tokenizer=tokenizer,model="apple/OpenELM-3B-Instruct", trust_remote_code=True)
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()
    summary = summarizer(text, num_beams,do_sample=False)

    print(summary)
def main():
    input_file="./text1.txt"
    print(input_file)
    num_beams=5
    summarize_text(input_file,num_beams)

if __name__ == '__main__':
    main()
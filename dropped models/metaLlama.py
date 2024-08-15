import torch
from transformers import pipeline
from transformers import AutoTokenizer
from huggingface_hub import login

# 24.74 gbs

def summarize_text(input_file,num_beams):
    
    login(token = "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")

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    summarizer = pipeline("summarization",tokenizer=tokenizer, model="meta-llama/Llama-2-7b-chat-hf")
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
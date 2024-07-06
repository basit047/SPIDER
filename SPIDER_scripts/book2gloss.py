
from nltk.stem import PorterStemmer # type: ignore
from nltk.tokenize import word_tokenize # type: ignore
from nltk import pos_tag # type: ignore
from collections import Counter
import re
import pandas as pd
import math
from transformers import AutoTokenizer, logging
import argparse
import os
import PyPDF2

# avoid warning: Special tokens has been added...
logging.set_verbosity_error()

        
def convert_pdf(pdf_path, output_path):
    '''Converts PDF to TXT file'''

    print("Converting PDF to TXT...")
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as text_file:
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text_file.write(page.extract_text())


def filter_tags(input_tag):
    '''A filter for tags'''

    tag_filter = ["NN", "NNS"]
    token, tag = input_tag

    return tag in tag_filter # if the tag is in tags list, it will return True


def stem_word(word):
    '''A function to perform stemming of given word'''

    global main_dict
    stemmed_word = ps.stem(word)
    main_dict[stemmed_word] = word.lower()
    return stemmed_word

    
def extract_nouns(input_text):
    '''Main SPIDER function to extract tokens from text'''

    cleaned_text = re.sub('[^a-zA-Z]', ' ', input_text)

    # tokenize text
    tokenized_text = word_tokenize(cleaned_text)
    
    # pos-tagging of stemmed text
    pos_tags = pos_tag(tokenized_text)
    
    # filter out non-Noun tags
    filtered_tags = list(filter(filter_tags, pos_tags))
    
    if len(filtered_tags) <= 0:
        return []

    tokens, tags = zip(*filtered_tags) #unpacking tokens and tags
    
    # stemming of text
    stemmed_tokens = list(map(stem_word, tokens))

    return stemmed_tokens


def get_tokens(txt_path):
    '''A function to apply SPIDER and to get tokens for given file'''

    print("Extracting data from file...")
    tokens = []

    with open(txt_path, encoding='utf-8') as file:
        for index, line in enumerate(file):
            temp_tokens = extract_nouns(input_text = line)
            tokens.extend(temp_tokens)
    return tokens


def filter_data(tokens):
    '''Filters out common tokens from the data'''

    print("Filtering out data...")
    frequencies_tokens = dict(Counter(tokens))
    df = pd.DataFrame({"tokenized_word": frequencies_tokens.keys(), "frequency": frequencies_tokens.values()})
    df = df.sort_values(by = ["frequency"])
    from_index = math.floor(len(df) * 0.90)
    return df.iloc[from_index:, :]


def tokenize_data(df):
    '''Applies phi3 tokenizer to the data and returns a dataframe with additional statistics'''

    print("Tokenizing data...")

    global main_dict

    data = []
    nouns = df["tokenized_word"].values

    for noun in nouns:
        word = main_dict[noun]  
        tokens = tokenizer.tokenize(noun)

        num_tokens = len(tokens)
        word_length = len(noun)

        data.append({"word": word, "tokens": tokens, "num_tokens": num_tokens, "word_length": word_length})

    df2 = pd.DataFrame(data)
    df2["ratio"] = df2["num_tokens"] / df2["word_length"]

    return df2


def get_unique_terms(df, min=0, max=1):
    '''Function to extract unique terms using statistics and the threshold value'''

    print("Extracting unique terms...")
    unique_terms = df.loc[df['ratio'] <= max].copy()
    unique_terms = unique_terms.loc[unique_terms['ratio'] > min]
    return unique_terms["word"].copy()


def apply_model(txt_path, output_path):
    '''Main pipeline of the model'''

    tokens = get_tokens(txt_path)
    df = filter_data(tokens)
    df = tokenize_data(df)

    unique_terms_df = get_unique_terms(df, min=0, max=0.4)
    unique_terms_df.to_csv(output_path, index=False)
    print(f"Success! Data is stored in file: {output_path}")


def check_file(path, type):
    '''Checks if the file exists at the location and has expected extension'''

    if os.path.exists(path):
        if not path.endswith(type):
            print(f"Please provide a valid file! (e.g. example.{type})")
            return False
    else:
        print("File not found! Please check if the path is correct.")
        return False
        
    return True


def cleanup(txt_path):
    '''Removing unnecessary files...'''

    if os.path.exists(txt_path):
        os.remove(txt_path)


# variables
ps = PorterStemmer()
model_path = "microsoft/Phi-3-mini-128k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
main_dict = {}
output_dir = "data"
txt_path = os.path.join(output_dir, "data.txt")
output_path = os.path.join(output_dir, "glossary.csv")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, help='Path to the input file', required=True)
    parser.add_argument('-t', choices=['pdf', 'txt'], default='pdf', help='Set input file type (default: pdf)')

    args = parser.parse_args()

    # create or check existance of output directory
    os.makedirs(output_dir, exist_ok=True)

    if args.t == "pdf":
        if not check_file(args.f, "pdf"):
            return
        
        convert_pdf(args.f, txt_path)
        apply_model(txt_path, output_path)
        cleanup(txt_path)

    elif args.t == "txt":
        if not check_file(args.f, "txt"):
            return
        
        apply_model(args.f, output_path)

if __name__ == "__main__":
    main()
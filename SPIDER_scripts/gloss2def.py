import pandas as pd
import argparse
import requests
import os

def convert_df_to_string(df):
    '''Converts a dataframe of words into a sinlge string'''
    if len(df) > 1:
        csv_string = ', '.join(df)
        
    return csv_string


def get_definition(data, model_name, user_prompt,  is_empty_request = False):
    '''Fetches definition from the LLM'''

    # data
    url = "http://localhost:11434/api/generate"
    
    headers = {"Content-Type": "application/json"}

    user_prompt = f"I have a book that is related to {user_prompt}, can you extract the unique words, give one line definition of the unique words and please remove the words which are not in context of {user_prompt} from this list:  {data}"
    query = {
    "model": model_name,
    **({"prompt": user_prompt, "keep_alive": "10m", "stream": False} if not is_empty_request else {})
}
    
    # http request
    response = requests.post(url, headers=headers, json=query)

    # response
    if response.status_code == 200:
        result = response.json()
        return result["response"]
    else:
        return f"Error {response.status_code}!"
    

def SPIDER_definitions(glossary_path, model_name, definitions_path, user_prompt):
    '''A function that serves as a pipeline for definition extraction'''

    print("Fetching file...")
    df = pd.read_csv(glossary_path)
    data = convert_df_to_string(df)

    print("Generating definitions...(grab a cup of coffee, this may take a while)")
    defs = get_definition(data, model_name, user_prompt)

    with open(definitions_path, "w") as file:
        file.writelines(defs)

    print(defs)
    print(f"Definitions are stored in file: {definitions_path}")


# variables
output_dir = "data"
default_glossary_path = os.path.join(output_dir, "glossary.csv")
definitions_path = os.path.join(output_dir, "definitioins.txt")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, help='Path to the glossary file generated from SPIDER', default=default_glossary_path)
    parser.add_argument('-m', type=str, choices=['phi3', 'mistral'], default='phi3', help='Model to use for generating definitions (default: phi3)')
    parser.add_argument('-p', type=str, help='Prompt for user input')

    args = parser.parse_args()

    SPIDER_definitions(args.f, args.m, definitions_path, args.p)

    

if __name__ == "__main__":
    main()
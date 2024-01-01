import sys
import pandas as pd
from transformers import pipeline
import os
from bs4 import BeautifulSoup
import unicodedata
import re



def preprocess(text):
    text = text.lower()

    if text:
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()

        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    return text


def predict_text(text):
    sentiment_analysis = pipeline("text-classification",model="/project/amazon_model")
    text = preprocess(text)
    result = sentiment_analysis(text)
    return pd.DataFrame([{"text": text, "sentiment": result[0]['label'], "confidence": result[0]['score']}])

def predict_csv(file_path):
    sentiment_analysis = pipeline("text-classification",model="/project/amazon_model")
    df = pd.read_csv(file_path)
    results = []
    for index, row in df.iterrows():
        text = preprocess(row['text'])
        result = sentiment_analysis(text)
        results.append({"text": row['text'], "sentiment": result[0]['label']})

    return pd.DataFrame(results)

def main():
    
    if len(sys.argv) != 2:
        print("Usage: python my_script.py 'text to process' or python my_script.py /path/to/csv/file.csv")
        sys.exit(1)

    input_arg = sys.argv[1]
    output_directory = '/project/output'
    os.makedirs(output_directory, exist_ok=True)

    output_csv_path = '/project/output/result.csv'

    if input_arg.endswith('.csv'):
        result_df = predict_csv(input_arg)
    else:
        result_df = predict_text(input_arg)

    result_df.to_csv(output_csv_path, mode='a', header=False, index=False,)



if __name__ == "__main__":
    main()
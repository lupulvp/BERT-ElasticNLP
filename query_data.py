import os
import sys
import torch
from transformers import AutoTokenizer, AutoModel
from elasticsearch import Elasticsearch
from dotenv import load_dotenv


def main(query, number_of_results):
    load_dotenv()
    data_index = os.getenv("DATA_INDEX")
    es_host = os.getenv("ES_HOST")

    if not data_index or not es_host:
        raise ValueError("Environment variables DATA_INDEX or ES_HOST are not set.")  # noqa

    print("Connecting to the Elasticsearch cluster...")
    es = Elasticsearch([es_host], verify_certs=False)
    if not es.ping():
        raise ConnectionError("Failed to connect to Elasticsearch.")

    try:
        print('Load the BERT tokenizer and model')
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModel.from_pretrained('bert-base-uncased')
    except Exception as e:
        print(f"Error during initialization: {e}")
        raise e

    try:
        print('Define a query text and convert it to a dense vector using BERT')
        inputs = tokenizer(query, return_tensors='pt', padding=True, truncation=True)  # noqa
        with torch.no_grad():
            output = model(**inputs).last_hidden_state.mean(dim=1).squeeze(0).numpy()  # noqa
        query_vector = output.tolist()

    except Exception as e:
        print(f"An error occurred: {e}")
        raise e

    try:
        print('Define the Elasticsearch KNN search')
        search = {
            "knn": {
                "field": "embedding",
                "query_vector": query_vector,
                "k": number_of_results,
                "num_candidates": 100
            },
            "fields": ["text"]
        }

        print('Perform the KNN search and print the results')
        response = es.search(index=data_index, body=search)

        print(f"\n\nTop {number_of_results} results for query `{query}`:")
        for hit in response['hits']['hits']:
            print(f"\n- {hit['_source']['text']}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query_data.py 'query text' number_of_results")
        sys.exit(1)

    query_text = sys.argv[1]
    results_count = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    main(query_text, results_count)

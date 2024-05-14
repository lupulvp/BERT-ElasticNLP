import os
import torch
from transformers import AutoTokenizer, AutoModel
from elasticsearch import Elasticsearch, helpers
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

DEFAULT_DATA_INDEX = "data-index"
DEFAULT_ES_HOST = "http://localhost:9200"


def create_index(es, data_index):
    if es.indices.exists(index=data_index):
        print("Deleting the index...")
        es.indices.delete(index=data_index)

    print("Creating the index...")
    mapping = {
        'properties': {
            'embedding': {
                'type': 'dense_vector',
                'dims': 768,
                'index': 'true',
                "similarity": "cosine"
            }
        }
    }
    es.indices.create(index=data_index, body={'mappings': mapping})


def main():
    data_index = os.getenv("DATA_INDEX") or DEFAULT_DATA_INDEX
    es_host = os.getenv("ES_HOST") or DEFAULT_ES_HOST
    es_batch_size = os.getenv("ES_BATCH_SIZE", 10)

    if not data_index or not es_host:
        raise ValueError(
            "Environment variables DATA_INDEX or ES_HOST are not set.")

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
        exit(1)

    try:
        print("Creating the Elasticsearch index...")
        create_index(es, data_index)
    except Exception as e:
        print(f"Error handling Elasticsearch index: {e}")
        raise e

    try:
        print("Loading the data and indexing it in Elasticsearch...")
        imdb_dataset = load_dataset("imdb", split="train")
        actions = []

        print(f"Indexing {len(imdb_dataset)} documents")
        for i, item in enumerate(imdb_dataset):
            text = item.get('text', None)
            label = item.get('label', -1)

            if text is None:
                continue

            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)  # noqa
            with torch.no_grad():
                output = model(**inputs).last_hidden_state.mean(dim=1).squeeze(0).numpy()  # noqa

            action = {
                "_index": data_index,
                "_source": {
                    'text': text,
                    'label': label,
                    'embedding': output.tolist(),
                }
            }
            actions.append(action)

            if len(actions) >= es_batch_size:
                helpers.bulk(es, actions)
                actions = []  # clear actions after processing
                print(f"Indexed {i+1} documents")

        if actions:  # index remaining documents
            helpers.bulk(es, actions)
    except Exception as e:
        print(f"Error during data processing and indexing: {e}")
        raise e


if __name__ == "__main__":
    main()

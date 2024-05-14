import os
import torch
from transformers import AutoTokenizer, AutoModel
from elasticsearch import Elasticsearch
from datasets import load_dataset

from dotenv import load_dotenv
load_dotenv() 
data_index = os.getenv("DATA_INDEX")
es_host = os.getenv("ES_HOST")

print("Connecting to the Elasticsearch cluster...")
es = Elasticsearch([es_host], verify_certs=False)

print('Load the BERT tokenizer and model')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Check if the index exists, delete it if it does
if es.indices.exists(index=data_index):
    print("Deleting the index...")
    es.indices.delete(index=data_index)

# if the index does not exist, create it with the defined mapping
if not es.indices.exists(index=data_index):
    print("Creating the index...")

    # Define the mapping for the dense vector field
    mapping = {
        'properties': {
            'embedding': {
                'type': 'dense_vector',
                'dims': 768,  # the number of dimensions of the dense vector
                'index': 'true',
                "similarity": "cosine"
            }
        }
    }
    es.indices.create(index=data_index, body={'mappings': mapping})

    print("Loading the data and indexing it in Elasticsearch...")
    imdb_dataset = load_dataset("imdb", split="train")

    inserted_count = 0
    print(f"Indexing {len(imdb_dataset)} documents")
    for item in imdb_dataset:
        # print(item)
        text = item.get('text', None)
        label = item.get('label', -1)

        if text is None:
            continue

        inputs = tokenizer(text, return_tensors='pt',
                           padding=True, truncation=True)
        output = None
        with torch.no_grad():
            output = model(**inputs).last_hidden_state.mean(dim=1).squeeze(0).numpy()  # noqa

        item_body = {
            'text': text,
            'label': label,
            'embedding': output.tolist(),
        }

        es.index(index=data_index, body=item_body)

        inserted_count += 1
        if inserted_count % 1000 == 0:
            print(f"Indexed {inserted_count} documents")
else:
    print("The index already exists. Skipping the data loading and indexing.")

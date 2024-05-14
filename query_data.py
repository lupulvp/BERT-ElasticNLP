import os
import torch
from transformers import AutoTokenizer, AutoModel
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

load_dotenv() 
data_index = os.getenv("DATA_INDEX")
es_host = os.getenv("ES_HOST")

print("Connecting to the Elasticsearch cluster...")
es = Elasticsearch([es_host], verify_certs=False)

print('Load the BERT tokenizer and model')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')


query = "Italian mini-series"
number_of_results = 2

print('Define a query text and convert it to a dense vector using BERT')
inputs = tokenizer(query, return_tensors='pt', padding=True, truncation=True)
output = None
with torch.no_grad():
    output = model(**inputs).last_hidden_state.mean(dim=1).squeeze(0).numpy()
query_vector = output.tolist()

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

print(f"Top {number_of_results} results:")
for hit in response['hits']['hits']:
    print(f"\n- {hit['_source']['text']}")

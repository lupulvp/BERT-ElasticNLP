# BERT-Enabled Elasticsearch

This project integrates Elasticsearch with BERT (Bidirectional Encoder Representations from Transformers) for creating a sophisticated text search platform. It includes scripts for indexing text data using BERT embeddings and querying them with natural language questions to retrieve relevant documents.

## Prerequisites

- Docker
- Docker Compose
- Python 3.8+
- Libraries: `torch`, `transformers`, `elasticsearch`, `datasets`, `python-dotenv`

## Setup
### 1. Clone the repo

```bash
git clone https://github.com/lupulvp/BERT-ElasticNLP.git
cd BERT-ElasticNLP
```

### 2. Environment Variables:
Copy the .env-template file to a new file named .env and update the environment variables if necessary:

```bash
cp .env-template .env
```

### 3. Install requirements
```bash
pip install -r requirements.txt
```

### 4. Elasticsearch and Kibana:
Start the services using Docker Compose:

```bash
docker-compose up -d
```

### 5. Index Data:
Run the load_data.py script to load and index the data:

```bash
python load_data.py
```

### 6. Query Data:
Use the query_data.py script to query the indexed data. Example:

```bash
python query_data.py "example query text" 5
```


## Architecture
- **Elasticsearch**: Used for storing and searching text documents. Configured to run in a Docker container.
- **BERT**: Utilized to convert text into dense vectors that Elasticsearch can use to perform similarity searches.
- **Kibana**: Provides a web interface for managing and visualizing data in Elasticsearch.

## Scripts
- `load_data.py`: Loads the IMDb dataset, processes it with BERT to generate embeddings, and indexes these embeddings in Elasticsearch.
- `query_data.py`: Performs a search in Elasticsearch using a BERT-generated embedding from a query text.


## Docker Compose
The `docker-compose.yml` file contains configurations for running Elasticsearch and Kibana in Docker containers. Elasticsearch is set up as a single-node cluster with specific configurations for development purposes.

## Troubleshooting
- Ensure Docker services are running correctly.
- Verify environment variables in the .env file.
- Check network settings if there are connectivity issues between the Python scripts and Elasticsearch.

## Official documentation
- [Elasticsearch Documentation](https://www.elastic.co/guide/index.html)
- [Hugging Face BERT](https://huggingface.co/docs/transformers/en/model_doc/bert)

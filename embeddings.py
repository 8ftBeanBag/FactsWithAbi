import openai
from decouple import config
import pinecone


INDEX = "shakespeare"
def connect():
    openai.organization = config('OPEN_AI_ORG')
    openai.api_key = config('OPEN_AI_KEY')

    pinecone.init(api_key=config("PINECONE_KEY"), environment=config("PINECONE_ENV"))


def create_idx():
    idxs = pinecone.list_indexes()
    if INDEX not in idxs:
        pinecone.create_index(INDEX, dimension=1536, metric="cosine")
        pinecone.describe_index(INDEX)

def save_embeddings(embeddings):
    index = pinecone.Index(INDEX)
    index.upsert(embeddings)

def query_idx(vec):
    index = pinecone.Index(INDEX)
    res = index.query(vector=vec, top_k=10000, include_values=True, namespace="")
    return list(map(lambda x: x['id'], res.matches))

def delete_idx():
    pinecone.delete_index(INDEX)

def get_idx_stats():
    index = pinecone.Index(INDEX)
    print(pinecone.describe_index(INDEX))
    print(index.describe_index_stats())

def create_embedding(text):
    embedding = openai.Embedding.create(input=text, model="text-embedding-ada-002")["data"][0]["embedding"]
    return embedding

if __name__=="__main__":
    connect()
    print(get_idx_stats())
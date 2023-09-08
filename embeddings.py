import pinecone
import openai
from decouple import config



INDEX = "shakespeare"
def connect():
    openai.organization = config('OPEN_AI_ORG')
    openai.api_key = config('OPEN_AI_KEY')

    pc_key = config("PINECONE_KEY")
    pc_env = config('PINECONE_ENV')
    pinecone.init(api_key=pc_key, environment=pc_env)
    idxs = pinecone.list_indexes()
    if INDEX not in idxs:
        pinecone.create_index(INDEX, dimension=1536, metric="euclidean")
        pinecone.describe_index(INDEX)

def save_embeddings(embeddings):
    index = pinecone.Index(INDEX)
    index.upsert(embeddings)

def query_idx(vec):
    index = pinecone.Index(INDEX)
    return index.query(vector=vec, top_k=3, include_values=True)

def get_idx_stats():
    index = pinecone.Index(INDEX)
    print(index.describe_index_stats())

def create_embedding(text):
    embedding = openai.Embedding.create(input=text, model="text-embedding-ada-002")["data"][0]["embedding"]
    return embedding

if __name__=="__main__":
    connect()
    print(get_idx_stats())
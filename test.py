# test.py
import pinecone
from full import connect, INDEX, get_embeddings_model

if __name__=="__main__":
    # Connect to Pinecone and get index
    connect()
    index = pinecone.Index(INDEX)

    # Get model and a single embedding
    model = get_embeddings_model()
    vec = model.embed_query("I am a test string")

    # Query the index and print the result
    res = index.query(vector=vec, top_k=3, include_values=True, namespace="")
    print(list(map(lambda x: x['id'], res.matches)))
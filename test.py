# test.py
import pinecone
import numpy as np
from full import connect, INDEX, get_embeddings_model

if __name__=="__main__":
    # Connect to Pinecone and get index
    connect()
    index = pinecone.Index(INDEX)

    # Get model and a single embedding
    model = get_embeddings_model()
    vec = model.embed_query("Darkness fell heavy upon my house")
    negative_vec = np.negative(vec)
    # Query the index and print the result
    res = index.query(vector=list(negative_vec), top_k=3, include_values=True, namespace="")
    print(list(map(lambda x: x['id'], res.matches))[0])
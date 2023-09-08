from shakespeare import scrape_shakespeare
from embeddings import connect, create_embedding, save_embeddings, query_idx
import json

def store_embeddings():
    connect()
    #corpus = scrape_shakespeare().split("\n")
    #print(corpus[1000])
    text = "Towards thee I'll run, and give him leave to go."
    #vec = create_embedding(text)
    with open("vec.txt", "r") as f:
        vec = json.loads(f.read())
    save_embeddings([(text,vec)])

def query_embeddings(text):
    connect()
    #vec = create_embedding(text)
    # with open("vec_res1.txt", "w") as f:
    #     f.write(json.dumps(vec))

    with open("vec_res1.txt", "r") as f:
        vec = json.loads(f.read())

    query_idx(vec)

if __name__=="__main__":
    # store_embeddings()
    print(query_embeddings("Towards thee I'll run, and give him leave to go."))
    
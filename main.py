from shakespeare import scrape_shakespeare
from embeddings import connect, create_embedding, save_embeddings, query_idx, delete_idx, create_idx
import json
import time

def store_embeddings():
    connect()
    #delete_idx()
    #create_idx()
    corpus = scrape_shakespeare().split("\n")
    data = []
    print(len(corpus[1600:2000]))
    filtered = filter(lambda x: len(x.strip().replace("\n", "")) > 4, corpus)
    for idx in range(0,100):
        text = next(filtered, None) + " " + next(filtered, None) + " " + next(filtered, None) + " " + next(filtered, None)
        text.replace("’", "'")

        if not text:
            break

        # Clean up text
        text.strip().replace("\n", '')
        if len(text) and text[0] == "Enter":
            continue

        vec = create_embedding(text)
        data.append((text, vec))
        time.sleep(1)
    # with open("vec.txt", "r") as f:
    #     vec = json.loads(f.read())
    data = list(map(lambda x: (x[0].replace("’", "'").replace("‘", "'").replace("—", "-"), x[1]), data))
    save_embeddings(data)

def query_embeddings(text):
    connect()
    vec = create_embedding(text)
    # with open("vec_res1.txt", "w") as f:
    #     f.write(json.dumps(vec))

    # with open("vec_res1.txt", "r") as f:
    #     vec = json.loads(f.read())

    return query_idx(vec)

if __name__=="__main__":
    # store_embeddings()
    x = query_embeddings("Light")
    print(x[0])
    
    
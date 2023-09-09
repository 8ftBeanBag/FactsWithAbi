from shakespeare import scrape_shakespeare
from embeddings import connect, create_embedding, save_embeddings, query_idx, delete_idx, create_idx
import json
import time
from langchain.embeddings import OpenAIEmbeddings
from decouple import config
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import pinecone
model_name = 'text-embedding-ada-002'



def store_embeddings():
    pinecone.init(api_key=config("PINECONE_KEY"), environment=config("PINECONE_ENV"))

    tiktoken.encoding_for_model('gpt-3.5-turbo')
    tokenizer = tiktoken.get_encoding('cl100k_base')
    # create the length function
    def tiktoken_len(text):
        tokens = tokenizer.encode(
            text,
            disallowed_special=()
        )
        return len(tokens)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
    )

    embeddings_model = OpenAIEmbeddings(
        model=model_name, 
        openai_api_key=config("OPEN_AI_KEY"), 
        max_retries=1,
        skip_empty=True,
        show_progress_bar=True)
    corpus = scrape_shakespeare().replace("’", "'").replace("‘", "'").replace("—", "-").replace("™", "TM")
    record_texts = text_splitter.split_text(corpus)
    # with open("vec.txt", "r") as f:
    #     vec = json.loads(f.read())

    TOKEN_BATCH = 1500
    while len(record_texts):
        tokens = 0
        batch = []
        while tokens < TOKEN_BATCH: 
            rec = record_texts.pop()
            if len(rec) > 512 or len(rec) != len(rec.encode()):
                print("Bad string, bad")
                continue 
            tokens += len(rec)
            batch.append(rec)
        embeddings = embeddings_model.embed_documents(batch)
        save_embeddings(zip(batch, embeddings))
        time.sleep(0.5)

def query_embeddings(text):
    connect()
    vec = create_embedding(text)
    # with open("vec_res1.txt", "w") as f:
    #     f.write(json.dumps(vec))

    # with open("vec_res1.txt", "r") as f:
    #     vec = json.loads(f.read())

    return query_idx(vec)

if __name__=="__main__":
    store_embeddings()
    x = query_embeddings("Dark")
    print(x[len(x)-1])
    
    
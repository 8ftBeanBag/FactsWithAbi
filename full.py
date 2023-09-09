import pinecone
from decouple import config
from bs4 import BeautifulSoup
import requests
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import time

MODEL_NAME = 'text-embedding-ada-002'
URL = "https://www.gutenberg.org/cache/epub/100/pg100.txt"
INDEX = "shakespeare"

def connect():
    pinecone.init(api_key=config("PINECONE_KEY"), environment=config("PINECONE_ENV"))
    idxs = pinecone.list_indexes()
    if INDEX not in idxs:
        pinecone.create_index(INDEX, dimension=1536, metric="cosine")
        pinecone.describe_index(INDEX)

def scrape_shakespeare():
    r = requests.get(URL)

    soup = BeautifulSoup(r.content, 'html.parser')
    corpus = soup.get_text()
    with open('shakespeare.txt', 'w') as f:
        f.write(corpus)

def read_shakespeare():
    with open("shakespeare.txt", "r") as f:
        corpus = f.read()
    return corpus

def get_text_splitter():
    tiktoken.encoding_for_model('gpt-3.5-turbo')
    tokenizer = tiktoken.get_encoding('cl100k_base')
    # create the length function
    def tiktoken_len(text):
        tokens = tokenizer.encode(
            text,
            disallowed_special=()
        )
        return len(tokens)
    return RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
    )

def get_embeddings_model():
   return OpenAIEmbeddings(
        model=MODEL_NAME, 
        openai_api_key=config("OPEN_AI_KEY"), 
        max_retries=1,
        skip_empty=True,
        show_progress_bar=True)

if __name__=="__main__":
    # Connect to Pinecone and get index
    connect()
    index = pinecone.Index(INDEX)

    # Get document
    # scrape_shakespeare()
    text = read_shakespeare()

    # Get text splitter object for splitting text into manageable and meaninful chunks
    text_splitter = get_text_splitter()

    # Get embeddings model for creating embeddings
    model = get_embeddings_model()

    # Clean text and split it up
    text.encode("ascii", "ignore")
    record_texts = text_splitter.split_text(text)

    TOKEN_BATCH = 15000
    while len(record_texts):
        tokens = 0  # Keeps track of teh number of tokens encountered
        batch = []  # Collects the texts to send in a batch to the database

        # coninue to get records until the batch limit is reached
        while tokens < TOKEN_BATCH: 
            # Get the next record in the list
            rec = record_texts.pop()
            
            # If the string is too long, Pinecone will throw an error. Plus we do not want
            # long strings to be returned due to spacing on the frontend.
            if len(rec) > 512 or len(rec) != len(rec.encode()):
                print("Bad string, bad")
                continue 
            
            # Update the amount of tokens we've used and save the rec to the batch
            tokens += len(rec)
            batch.append(rec)

        # Get the embeddings for the strings
        embeddings = model.embed_documents(batch)
        # Insert into the database
        index.upsert(zip(batch, embeddings))
        # Keeps the loop from exceeding rate limits
        time.sleep(0.1)
    
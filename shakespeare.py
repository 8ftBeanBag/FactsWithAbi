from bs4 import BeautifulSoup
import requests
url = "https://www.gutenberg.org/cache/epub/100/pg100.txt"

def scrape_shakespeare():
    # r = requests.get(url)

    # soup = BeautifulSoup(r.content, 'html.parser')
    # corpus = soup.get_text()
    # with open('shakespeare.txt', 'w') as f:
    #     f.write(corpus)

    with open("shakespeare.txt", "r") as f:
        corpus = f.read()
    return corpus

if __name__=="__main__":
    print(scrape_shakespeare())
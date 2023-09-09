from flask import Flask, jsonify, request
from main import query_embeddings
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/', methods=["POST"])
@cross_origin()
def get_fact():
    print(request.get_json())
    text = request.get_json().get("text")
    if not text.strip():
        return jsonify ({"response": "Oops, no white space only allowed."})
    res = query_embeddings(text)
    return jsonify({"response": res[len(res)-1]})

if __name__=="__main__":
    app.run(host="0.0.0.0", port=8080)
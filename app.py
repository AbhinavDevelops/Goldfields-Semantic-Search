from flask import Flask, request, jsonify, render_template
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load data and model
df = pd.read_excel('data.XLSX')
desc = df["Material Description"].to_list()

bin_loc = df["Bin Location"].to_list()
sloc = df["SLoc"].to_list()
material = df["Material"].to_list()
bUN = df["BUn"].to_list()
stock_qty = df["Storage Location Stock Qty"].to_list()

embeddings_file = 'embeddings.npy'
index_file = 'faiss_index.idx'

model = SentenceTransformer('all-MiniLM-L6-v2')

if os.path.exists(embeddings_file) and os.path.exists(index_file):
    embeddings = np.load(embeddings_file)
    index = faiss.read_index(index_file)
else:
    embeddings = model.encode(desc)
    embeddings = np.array(embeddings).astype('float32')
    np.save(embeddings_file, embeddings)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_file)


@app.route('/')
def landing():
    return render_template('index2.html')


@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400

    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding).astype('float32'), 50)

    results = []
    for idx in I[0]:
        results.append({
            "description": desc[idx],
            "bin_location": bin_loc[idx],
            "sloc": sloc[idx],
            "material": material[idx],
            "bUN": bUN[idx],
            "stock_qty": stock_qty[idx]
        })

    return jsonify({"results": results})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

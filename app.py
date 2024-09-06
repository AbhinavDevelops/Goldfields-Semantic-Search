from flask import Flask, request, jsonify
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load data and model
df = pd.read_excel('data.XLSX')
desc = df["Material Description"].to_list()

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


@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400

    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding).astype('float32'), 50)
    results = [desc[idx] for idx in I[0]]

    return jsonify({"results": results})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

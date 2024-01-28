import pickle
import flask
import pandas as pd
from flask import jsonify, request
from scipy import sparse

app = flask.Flask(__name__)
model = None
matrix = None
id = None
filename = 'finalized_model.sav'
recommendations = 10

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    model = pickle.load(open(filename, 'rb'))


    matrix = sparse.load_npz("yourmatrix.npz")
    id = pd.read_csv('saved_id.csv')
    id['product_id'] = id['product_id'].astype(str)
    idcode = data["input"][0]
    product_id = id[id['product_id'] == idcode].index[0]
    distances, indices = model.kneighbors(matrix[product_id], n_neighbors=recommendations + 1)

    indices_list = indices.squeeze().tolist()
    recom_list = []
    for ind_dist in indices_list:
        dist=id.loc[ind_dist]
        recom_list.append({'id' : dist.product_id, 'Title' : dist.product_name})

    distances_list = distances.squeeze().tolist()
    indices_distances = list(zip(recom_list, distances_list))

    indices_distances_sorted = sorted(indices_distances, key=lambda x: x[1], reverse=False)
    indices_distances_sorted = indices_distances_sorted[1:]


    return jsonify(indices_distances_sorted)


if __name__ == "__main__":


    app.run()

"""
The main web api for the subject identifier package.

Takes POST requests with the data of a paper title and abstract, returns the preposed subject classifcations.
"""

import pandas as pd
import joblib
import os
from flask import Flask, request, jsonify

from subject_identifier import identify_subjects

app = Flask(__name__)

@app.route('/', methods=['POST'])
def recommend():
    data = request.json
    title=data.get('input_title', None)
    abstract=data.get('input_abstract', None)
    results = identify_subjects(input_title=title, input_abstract=abstract)
    results_dict = results.to_dict()
    results_json = jsonify(results_dict)
    return jsonify(results.to_dict())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
    #app.run(host='0.0.0.0', port=8080,debug=True)
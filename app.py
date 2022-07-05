import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import lightgbm as lgbm
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
import category_encoders as ce
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.metrics import roc_auc_score, ConfusionMatrixDisplay, roc_curve, precision_recall_curve,classification_report,  accuracy_score, f1_score, recall_score, precision_score
import pickle
from flask import Flask, request, url_for, redirect, render_template, jsonify

warnings.filterwarnings("ignore")

with open(f'model.sav', 'rb') as f:
    model = pickle.load(f)
    


# Initalise the Flask app
app = Flask(__name__, template_folder='templates')

cols = ['NUMSITUA', 'EDAD', 'CAPIASEG', 'PNETA_DECESOS', 'PERCEN_NEG',
       'PRECIPITACION', 'TMAX', 'SEXO', 'COBROPER', 'TIPOCALC',
       'MES_OBSERVACION']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = model.predict_proba(data_unseen)[:,1]
    return render_template('home.html',pred='Probabilidad de fuga: {}'.format(prediction))


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = model.predict_proba(data_unseen)[:,1]
    output = prediction.Label[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)

print("End of the program")
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


# Dataset
df_datos = pd.read_csv('datos_clean.csv')

df_datos['CAPIASEG']=df_datos['CAPIASEG'].astype(int)
df_datos['MES_OBSERVACION']= df_datos['MES_OBSERVACION'].astype(str)

data = df_datos

train = df_datos.drop('IS_BAJA', axis=1)
target = df_datos['IS_BAJA']

X_train, X_test, y_train, y_test = train_test_split(train, target, stratify=target, shuffle=True, test_size=0.20, random_state=42)
#comprobación de la estratificación
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))

continuous = ['NUMSITUA', 'EDAD', 'CAPIASEG','PNETA_DECESOS', 'PERCEN_NEG', 'PRECIPITACION', 'TMAX']
categorical = ['SEXO', 'COBROPER', 'TIPOCALC', 'MES_OBSERVACION']

continuous = train.select_dtypes(exclude=['object']).columns
categorical = list(set(X_train.columns)-set(continuous))


#1st Transformer
trf1 = ColumnTransformer([
    ('catboost',ce.CatBoostEncoder(),categorical)])

#2nd Transformer: Scaling
trf2 = ColumnTransformer([('scaler', StandardScaler(),slice(0,len(X_train.columns)+1))
                     ])

#3nd Transformer: SMOTE para tratar el desbalanceado de datos
over = SMOTE(sampling_strategy=0.2, random_state=42)

parametros = {'learning_rate': 0.01,
 'max_depth': 3,
 'n_estimators': 600,
 'subsample': 0.66}


#Model
model = lgbm.LGBMClassifier(**parametros)


pipe = Pipeline(steps=[
('trf1', trf1),
('trf2', trf2),
('over', over),
('model', model)
])

pipe.fit(X_train, y_train)

print("Model Trained")

preds = pipe.predict(X_test)
# precisión de las predicciones del modelo
model_accuracy = accuracy_score(list(y_test), preds)
f1_score= f1_score(y_true= y_test, y_pred=preds.astype(int))
roc_auc = roc_auc_score(y_true= y_test, y_score=pipe.predict_proba(X_test)[:, 1])
recall = recall_score(y_test, y_pred=preds)
print("Model Accuracy : ", model_accuracy)
print("f1_score : ", f1_score)
print("roc_auc :", roc_auc)
print("recall :", recall)

"""
# Save the model
filename = 'model.sav'
pickle.dump(pipe, open(filename, 'wb'))"""
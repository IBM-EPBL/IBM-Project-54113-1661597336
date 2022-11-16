import pickle

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app=Flask(__name__, template_folder='templates')
model=pickle.load(open('CKD.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/details.html') 
def details():
    return render_template('details.html')

@app.route('/awareness.html') 
def awareness(): 
    return render_template('awareness.html')

@app.route('/diagnosis.html') 
def diagnosis():
    return render_template('diagnosis.html')           


@app.route('/prediction.html')
def prediction():
    return render_template('prediction.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    
    input_features=[float(x) for x in request.form.values()]
    features_value=[np.array(input_features)]

    features_name=['blood_urea','blood glucose random','coronary_artery_disease','anemia','pus_cell','red_blood_cells','diabetesmellitus','pedal_edema']
    
    df=pd.DataFrame(features_value, columns=features_name)

    output=model.predict(df)
    
    if output == 1:
        return render_template('success.html')
    else:
        return render_template('failure.html')


if __name__ == '__main__':
    app.run(debug=True) 
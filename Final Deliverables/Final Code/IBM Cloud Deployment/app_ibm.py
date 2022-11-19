import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import requests
import pickle

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "QXCDaXxG__YG-rHuYkTQwfeatR70SW8MTP9Lr29ORWNB"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

model=pickle.load(open('CKD.pkl','rb'))

# NOTE: manually define and pass the array(s) of values to be scored in the next line

app=Flask(__name__, template_folder='templates')

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

    #showing the prediction resultsin a UI# showing the prediction results in UI
    if output == 1:
        return render_template('success.html')
    else:
        return render_template('failure.html')
    
    


    payload_scoring = {"input_data":  [{"fields": [['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc','htn','dm','cad','appet','pe','ane']], "values": features_name}]}
    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/e0e99a97-2d3d-482b-bedf-d1a5f97dfdc4/predictions?version=2022-11-12', json=payload_scoring,
    headers={'Authorization': 'Bearer ' + mltoken})
    print("Scoring response")
    print(response_scoring.json())
if __name__ == '__main__':
    app.run(debug=True) 

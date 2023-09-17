from flask import Flask,render_template,request,jsonify
import pickle
import numpy as np
import json

app = Flask("Predictor",static_url_path='/static', static_folder='static')

dc_model = pickle.load(open("Models/dctree_model.pkl", 'rb'))
scaler = pickle.load(open("Models/scaler.pkl","rb"))
rfc_model = pickle.load(open("Models/rfc_model.pkl","rb"))
# nn_model = pickle.load(open("Models/nn_model.pkl","rb"))
lr_model = pickle.load(open("Models/lr_model.pkl","rb"))

def transform_data(inpt):
    locations = ['Los Angeles', 'New York', 'Miami', 'Chicago', 'Houston']
    gender = ["Female","Male"]
    inp = []

    for i in inpt:

        if i in locations:
            inp.append(locations.index(i)+1)
        
        elif i in gender:
            inp.append(gender.index(i))

        else:
            inp.append(i)


    inp = scaler.transform(np.array(inp).reshape(1,-1))
    return inp 



@app.route('/')
def index():
    return render_template("home.html")

@app.route('/dtree' , methods = ['POST'])
def predict_dtree():
    if request.method == 'POST':

        data = json.loads(request.data)
        age = data["age"]
        gender = data["gender"]
        location = data["location"]
        bill = data["bill"]
        subLen = data["subLen"]
        totalGB = data["totalGB"]
        
        inp = transform_data([age,gender,location,bill,subLen,totalGB])
        res = str(dc_model.predict(inp)[0])
        return {"sucess": True, "result" : res}
    

@app.route('/rfc' , methods = ['POST'])
def predict_rfc():
    if request.method == 'POST':

        data = json.loads(request.data)
        age = data["age"]
        gender = data["gender"]
        location = data["location"]
        bill = data["bill"]
        subLen = data["subLen"]
        totalGB = data["totalGB"]
        
        inp = transform_data([age,gender,location,bill,subLen,totalGB])
        res = str(rfc_model.predict(inp)[0])
        return {"sucess": True, "result" : res}
    

@app.route('/nn' , methods = ['POST'])
def predict_nn():
    if request.method == 'POST':

        data = json.loads(request.data)
        age = data["age"]
        gender = data["gender"]
        location = data["location"]
        bill = data["bill"]
        subLen = data["subLen"]
        totalGB = data["totalGB"]
        
        inp = transform_data([age,gender,location,bill,subLen,totalGB])
        res = nn_model.predict(inp)[0]
        res = str((res > 0.5).astype(int)[0])
        return {"sucess": True, "result" : res}
    
@app.route('/lr' , methods = ['POST'])
def predict_lr():
    if request.method == 'POST':

        data = json.loads(request.data)
        age = data["age"]
        gender = data["gender"]
        location = data["location"]
        bill = data["bill"]
        subLen = data["subLen"]
        totalGB = data["totalGB"]
        
        inp = transform_data([age,gender,location,bill,subLen,totalGB])
        res = str(lr_model.predict(inp)[0])
        return {"sucess": True, "result" : res}
        
    

app.run(debug=True,host='0.0.0.0',port=8080)





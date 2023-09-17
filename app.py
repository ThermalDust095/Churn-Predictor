from flask import Flask,render_template,request
import pickle
import numpy as np

app = Flask("Predictor")

dc_model = pickle.load(open("Models/dctree_model.pkl", 'rb'))
scaler = pickle.load(open("Models/scaler.pkl","rb"))

def transform_data(inpt):
    locations = ['Los Angeles', 'New York', 'Miami', 'Chicago', 'Houston']
    gender = ["Female","Male"]
    inp = []

    for i in inpt:

        if i in locations:
            print(locations.index(i)+1)
            inp.append(locations.index(a)+1)
        
        elif i in gender:
            inp.append(i)

        else:
            inp.append(i)


    inp = scaler.transform(np.array(inp).reshape(1,-1))
    return inp 



@app.route('/')
def index():
    return render_template("home.html")

@app.route('/dtree',methods=['GET'])
def predict():

    age = request.form.get('age')
    gender = request.form.get('gender')
    location = request.form.get('location')
    bill = request.form.get('bill')
    sub_len = request.form.get('sub_len')
    totalgb = request.form.get('total-gb')

    return {"result": [age,gender,location,bill,sub_len,totalgb]}


app.run(debug=True)





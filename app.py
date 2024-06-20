import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd 

# entrypoint of my flask file
app=Flask(__name__)

# Load the pickle file
reg_pickle_file=open('./regression.pkl','rb')
scalar_pickle_file=open('./scaling.pkl','rb')
regmodel=pickle.load(reg_pickle_file)

scalar=pickle.load(scalar_pickle_file)

# my home page
@app.route('/')
def home():
    return render_template('home.html')


# my prediction api 
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    # convert the incoming data to a list
    list_data = list(data.values())

    # convert it into a np array of required shape 1 row and 'feature' no of columns
    reshaped_array=np.array(list_data).reshape(1,-1)

    # standarize the data using the scalar object
    new_data=scalar.transform(reshaped_array)

    #predict the house price
    output=regmodel.predict(new_data)

    #print the predicted house price
    print(output[0])

    return jsonify(output[0])


@app.route('/predict',methods=['POST'])
def predict():
    # get the data from the POST request and convert it into a list
    data = [float(x) for x in request.form.values()]

    # reshape the data into 1 row and x no of columns
    reshaped_data = np.array(data).reshape(1,-1)

    # standardize the data
    final_input = scalar.transform(reshaped_data)

    # see the final_input here
    print(final_input)

    # prediction
    output=regmodel.predict(final_input)[0]

    # send it back to the browser
    return render_template("home.html", prediction_text="The predicted house price is {}".format(output))

if __name__=="__main__":
    app.run(debug=True)
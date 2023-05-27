from flask import Flask, request, render_template
import numpy as np
import pickle
app = Flask(__name__,template_folder="SITE",static_folder='./SITE/css')

# Default route set as 'home'
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    # Get the data from the POST request.
    features=[int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    # print("FINAL FEATURES = ",request.form.values())
    prediction = model.predict(final_features)
    data=prediction[0]
    print("PREDICTED VALUE = ",prediction[0])
    return render_template('index.html', predicted_text='{}'.format(data))

if(__name__=='__main__'):

    app.run(debug=True)
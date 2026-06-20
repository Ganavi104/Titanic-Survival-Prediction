
from flask import Flask, render_template, request
import pickle
import pandas as pd


app = Flask(__name__)


# Load saved model
model = pickle.load(open("titanic_model.pkl", "rb"))



@app.route("/")
def home():
    return render_template("index.html", prediction="")



@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Input features (same names as training)
        data = pd.DataFrame([{

            "pclass": int(request.form["pclass"]),
            "sex": int(request.form["sex"]),
            "age": float(request.form["age"]),
            "fare": float(request.form["fare"]),
            "sibsp": int(request.form["sibsp"]),
            "parch": int(request.form["parch"]),
            "embarked": int(request.form["embarked"])

        }])
        print(data.columns)
        print(data)


        # Predict
        result = model.predict(data)


        if result[0] == 1:
            prediction = "🚢 Passenger Survived"
        else:
            prediction = "❌ Passenger Did Not Survive"



        return render_template(
            "index.html",
            prediction=prediction
        )


    except Exception as e:

        return render_template(
            "index.html",
            prediction="Error: " + str(e)
        )



if __name__ == "__main__":
    app.run(debug=True)

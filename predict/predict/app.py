from flask import Flask, request, render_template
from run import TextPredictionModel

app = Flask(__name__)

artefacts_path = r"C:\Users\Arthur\OneDrive\Documents\cours\5A\From POC to prod\poc-to-prod-capstone\train\data\artefacts\2023-01-03-11-02-01"
model = TextPredictionModel.from_artefacts(artefacts_path)


@app.route('/')
def my_form():
    return render_template('my-form.html')


@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    prediction = model.predict([text], 1)[0][0]
    return prediction


if __name__ == "__main__":
    app.run(host='localhost', port=5000, debug=True)

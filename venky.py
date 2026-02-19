from  flask import Flask,jsonify
from joblib import load
app=Flask(__name__)
model=load("venkymodel")
@app.route("/")
def sam():
    return 'okay',200
@app.route("/pred")
def sams():
    v=[[0.623082, 0.0, 0.36, 0.0, 0.578947, 1.0, 0.758065, 0.444444, 0.233333, 0.9, 0.392764, 0.206413, 0.40526, 0.714286, 0.444444, 0.277778, 1.0,2.0, 0.0, 1.0]]
    e=model.predict(v)
    e=float(e[0])
    return jsonify(e,200)
if(__name__=="__main__"):
    app.run(debug=True,port=8000)
from flask import Flask, request
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('models/tree-model.pkl', 'rb'))


@app.route('/', methods=['POST'])
def predict():
    post_data = request.get_json()
    # Use of meta data keys e.g. post_data['date start']
    data = pd.DataFrame.from_dict(post_data['payload'], orient='columns')
    data = data[post_data['columns']]
    
    try:
        res = model.predict(data)
    except Exception as e:
        return {"error": str(e)}
        
    #model.n_classes_
    out_dict = {"pred": str(res)}
    return out_dict


if __name__ == '__main__':
    #app.run(debug=True)
    app.run(threaded=True, port=5000)

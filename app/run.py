import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Pie
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Message', engine)



# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extracting data needed for visuals
    
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    top_10_counts = df.iloc[:,4:].sum(axis=0,skipna=True).sort_values(ascending=False)[1:10]
    
    isrelated_counts = df.groupby('related').count()['id']
    

    # creating visuals
    
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Pie(
                    values=top_10_counts.values,
                    labels=top_10_counts.index
                )
            ],

            'layout': {
                'title': 'Distribution of Disaster Categories',
            }
        },
        {
            'data': [
                Bar(
                    x=['unrelated', 'related with 1 disaster type', 'related with 2 disaster types'],
                    y=isrelated_counts.values
                )
            ],

            'layout': {
                'title': 'Relative Messages Relations with Disaster',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Number of Disaster Relations of Message"
                }
            }
        }
    ]
    
    # encoding plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # rendering web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # saving user input in query
    query = request.args.get('query', '') 

    # using model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
from flask import Flask
from application.model import get_prediction
from flask import render_template, request
from application import app

@app.route('/', methods=['GET', 'POST'])
#@app.route('/index', methods=['GET', 'POST'])
@app.route('/input', methods=['GET', 'POST'])
def input():
    return render_template("input.html", title='Input company info')


@app.route('/output', methods=['GET', 'POST'])
def get_model_output():
    Company = request.args.get('Company')
    blurb, color = get_prediction(Company)
    return render_template('output.html', title='Results', 
                           blurb=blurb, color=color)

import sys
print('This error output', file=sys.stderr)
print('This standard output', file=sys.stdout)

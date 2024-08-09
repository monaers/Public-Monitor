from flask import Flask,session,render_template,redirect,Blueprint,request,flash
import os
import sys
from Analyse import FINAL
pb = Blueprint('page',__name__,url_prefix='/page',template_folder='templates')

@pb.route('/home')
def home():
    return render_template('index.html')

@pb.route('/analy',methods=['GET','POST'])
def analy():
    url = request.form['url']
    result = FINAL.wow(url)
    summery = result['summary']
    sentiment = result['sentiment']
    classification = result['classification']
    return render_template('analy.html',url=url,summery=summery,sentiiment=sentiment,classification=classification)
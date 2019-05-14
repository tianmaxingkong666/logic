# -*- coding: utf-8 -*-
import io
import json
import time
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import uuid

app = Flask(__name__)


# 加载模型
scaler = joblib.load('scaler.pkl')
clf = joblib.load('lr_credit.pkl')


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/api/predict/credit", methods=['POST'])
def team():
    file = request.files['file']
    jobid = uuid.uuid1().__str__()
    path = '{}.csv'.format(jobid)
    file.save(path)

    # csv 转换为 DataFrame
    df = pd.DataFrame(pd.read_csv(path,header=0))
    df.columns = ['Percentage', 'age', '30-59', 'DebtRatio', 'MonthlyIncome', 'Number_Open','90-','Number_Estate','60-89','Dependents']
    
	# 标准化数据
    X_test = scaler.transform(df)
	
	# 预测
    logit_scores_proba = clf.predict_proba(X_test)
  
    print({'res': logit_scores_proba[0][1]})
    return json.dumps({'res': logit_scores_proba[0][1]}, ensure_ascii=False)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9999)

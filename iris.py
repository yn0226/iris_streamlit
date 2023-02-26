#基本ライブラリ
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

#データセット読み込み
iris=load_iris()
df =pd.DataFrame(iris.data, columns=iris.feature_names)

#目標値の設定
df['target']=iris.target

#目標値の変数を花の名前に変更
df.loc[df['target'] ==0, 'target'] ='setosa'
df.loc[df['target'] ==1, 'target'] ='varsicolor'
df.loc[df['target'] ==2, 'target'] ='virginica'

#予測モデル構築
x= iris.data[:, [0,2]]  #0~1列目を入力変数にする
y= iris.target

#ロジスティック回帰
clf =LogisticRegression()
clf.fit(x,y)

#サイドバーの入力画面
st.sidebar.header('Input features')

sepalValue=st.sidebar.slider('sepal length (cm)',min_value=0.0, max_value=10.0,step=0.1)
petalValue=st.sidebar.slider('petal length (cm)',min_value=0.0, max_value=10.0,step=0.1)


#メイン画面に出す
st.title('Iris Classifier')
st.write('## Input value')


# インプットデータ（1行のデータフレーム）
value_df = pd.DataFrame([],columns=['data','sepal length (cm)','petal length (cm)'])
record = pd.Series(['data',sepalValue, petalValue], index=value_df.columns)
value_df = value_df.append(record, ignore_index=True)
value_df.set_index('data',inplace=True)

#入力値の値
st.write(value_df)

# 予測値のデータフレーム
pred_probs = clf.predict_proba(value_df)
pred_df = pd.DataFrame(pred_probs,columns=['setosa','versicolor','virginica'],index=['probability'])

st.write('## Prediction')
st.write(pred_df)

# 予測結果の出力
name = pred_df.idxmax(axis=1).tolist()
st.write('## Result')
st.write('このアイリスはきっと',str(name[0]),'です!')

#リモートリポジトリのURL
#https://github.com/yn0226/iris_streamlit.git




import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier



df=pd.read_csv('banka.csv')
df['metin']=df['metin'].str.lower()

stopwords=['ye','mı','mi','rica','istiyorum','fakat','lakin','ancak','acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri', 'birkaç', 'birşey', 'biz', 'bu', 'çok', 'çünkü', 'da', 'daha', 'de', 'defa', 'diye', 'eğer', 'en', 'gibi', 'hem', 'hep', 'hepsi', 'her', 'hiç', 'için', 'ile', 'ise', 'kez', 'ki', 'kim', 'mı', 'mu', 'mü', 'nasıl', 'ne', 'neden', 'nerde', 'nerede', 'nereye', 'niçin', 'niye', 'o', 'sanki', 'şey', 'siz', 'şu', 'tüm', 've', 'veya', 'ya', 'yani']

def sw(metin):
    kelimeler=metin.split(" ")
    for s in stopwords:
        if s in kelimeler:
            kelimeler.remove(s)
    return " ".join(kelimeler)

df['metin']=df['metin'].apply(sw)


cv=CountVectorizer(max_features=250)
x=cv.fit_transform(df['metin']).toarray()
y=df['kategori']

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.75,random_state=42)

rf=RandomForestClassifier()
model=rf.fit(x_train,y_train)
model.score(x_test,y_test)


cumle=st.text_area("Yapmak İstediğiniz İşlemi Yazınız")
ses=st.audio_input("Yapmak İstediğiniz İşlem")
if len(cumle)>1:
    tvektor=cv.transform([cumle]).toarray()
    sonuc=model.predict(tvektor)

    st.write(sonuc[0])
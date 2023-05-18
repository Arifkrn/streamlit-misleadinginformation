import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from streamlit_option_menu import option_menu


with st.sidebar:
    selected = option_menu('Menu Bar', ['Tes Kata', 'Dataset', 'Tentang Aplikasi'], default_index=0)
    
    #load model
    model_misleading_information = pickle.load(open('model_misleading_information.sav', 'rb'))

    #load dataset
    datatest = pd.read_csv('clean_data.csv')

    tfidf = TfidfVectorizer

    loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(pickle.load(open("new_selected_feature_tf-idf.sav", "rb"))))
    
if (selected=='Tes Kata'):
    #Judul
    st.title('Deteksi Misleading Information Tentang Vaksin Covid-19')

    clean_teks = st.text_input('Masukan Teks')

    misleadinginf_detection = ''

    if st.button('Hasil dari Deteksi'):
        predict_misleadinginf = model_misleading_information.predict(loaded_vec.fit_transform([clean_teks]))
    
        if(predict_misleadinginf == 0):
            misleadinginf_detection = 'Mengandung Misleading Information'
        else:
            misleadinginf_detection = 'Tidak Mengandung Misleading Information'

    st.success(misleadinginf_detection)
    st.info("Note : Masukan teks dari menu dataset pada kolom 'clean_teks' ")

if (selected=='Dataset'):
    st.header('Dataset')
    st.dataframe(datatest)
    st.info("Note : label bernilai '0' mengandung misleading information, sedangkan '1' tidak mengandung misleading information")
    
if (selected=='Tentang Aplikasi'):
    st.header('Tentang Aplikasi Deteksi Misleading Information')
    st.markdown("<p style='text-align: justify;'><br><br>Aplikasi Deteksi Misleading Information Tentang Vaksin Covid-19 merupakan aplikasi berbasis website yang dikembangkan dengan menggunakan bahasa pemrograman Python. Data yang telah diperoleh berasal dari Aplikasi Twitter berupa tweet tentang vaksin covid-19 pada tahun 2022. Tujuan dari aplikasi ini ialah mendeteksi informasi apakah mengandung misleading information ataukah tidak. Misleading Information sendiri memiliki arti informasi yang belum tentu benar. Dalam mendeteksi informasi yang disajikan sesuai atau tidak sesuai yakni dengan cara membandingkan informasi tersebut dengan media mainstream diantaranya : kompas.com, detik.com, kemenkes dan tempo.co serta CNN.com</p>", unsafe_allow_html=True)
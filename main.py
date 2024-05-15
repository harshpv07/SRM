import streamlit as st 
import pandas as pd 
import langchain 
import seaborn as sns 

st.text("Resume Parser")

with st.sidebar:
    txt_inp = st.text_input(label = "Enter your query")
    file_upload = st.file_uploader(label = "Upload your resume in PDF format")


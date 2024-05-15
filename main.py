import streamlit as st 
import pandas as pd 
import langchain 
import seaborn as sns 
from tools import * 
from streamlit_pdf_viewer import pdf_viewer 
from streamlit import session_state as ss 

st.text("Resume Parser")

with st.sidebar:
    txt_inp = st.text_input(label = "Enter your query")
    file_upload = st.file_uploader(label = "Upload your resume in PDF format")

    # parser_file(file_upload)


container_pdf, container_chat = st.columns([100, 100])


with container_pdf:
    
    if file_upload:
        binary_data = file_upload.getvalue()
        pdf_viewer(input=binary_data,
                   width=700)




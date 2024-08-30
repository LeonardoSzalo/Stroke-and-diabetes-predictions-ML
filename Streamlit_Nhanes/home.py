import streamlit as st
import intro
import demografico_streamlit  # Importa o primeiro script como um módulo
import cronicas_streamlit  # Importa o segundo script como um módulo
import predicoes_streamlit # Importa o terceiro script como um módulo

st.set_page_config(page_title="NHANES Model Prediction",layout="wide", page_icon = 'https://www.cdc.gov/nchs/images/nhanes/NHANES-Trademark.png?_=04691')
# Dicionário para armazenar as diferentes páginas
pages = {
    "About NHANES": intro,
    "Demographic analysis ": demografico_streamlit,
    "Chronic Disease Prevalence ": cronicas_streamlit,
    "Disease Prediction Using Machine Learning": predicoes_streamlit
    
}

# Sidebar para selecionar a página
st.sidebar.title("Menu")
selection = st.sidebar.radio("Go to", list(pages.keys()))

# Carregar a página correspondente
page = pages[selection]
page.run()  # Executa a função `run()` de cada script
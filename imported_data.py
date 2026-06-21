import streamlit as st
import pandas as pd


def imported_data_page():
    st.header("Dataset importado")

    uploaded_file = st.file_uploader("Selecione um arquivo CSV", type="csv")

    if uploaded_file is None:
        st.info("Selecione um arquivo.")
        return None, None

    df = pd.read_csv(uploaded_file)

    st.subheader("Pré-visualização")
    st.dataframe(df.head())

    numeric_columns = df.select_dtypes(include="number").columns

    if len(numeric_columns) < 2:
        st.warning("O dataset deve conter pelo menos duas variáveis numéricas.")
        return None, None

    x_column = st.selectbox("Variável independente (x)", numeric_columns)
    y_column = st.selectbox("Variável dependente (y)", numeric_columns)

    x = df[x_column].to_numpy()
    y = df[y_column].to_numpy()

    return x, y
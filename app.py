import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from synthetic_data import synthetic_data_page
from imported_data import imported_data_page
from utils import polynomial_to_latex


st.title("Regressão por Mínimos Quadrados")


# escolher fonte dos dados
st.sidebar.header("Fonte dos dados")

data_source = st.sidebar.radio(
    "Selecione a origem dos dados",
    (
        "Gerar dados sintéticos",
        "Importar dataset"
    )
)

if data_source == "Gerar dados sintéticos":
    x, y, beta_true = synthetic_data_page()
else:
    x, y = imported_data_page()


if x is not None and y is not None:
    # cálculo da regressão
    X = np.column_stack((np.ones_like(x), x))
    beta_pred, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ beta_pred

    # exibir resultados
    st.write("### Resultados do modelo")

    if data_source == "Gerar dados sintéticos":
        col1, col2 = st.columns(2)

        with col1:
            st.write("#### Valores reais")
            st.write(rf"$\beta_0 = {beta_true[0]:.4f}$")
            st.write(rf"$\beta_1 = {beta_true[1]:.4f}$")
            st.write(polynomial_to_latex(beta_true))

        with col2:
            st.write("#### Valores estimados")
            st.write(rf"$\hat{{\beta_0}} = {beta_pred[0]:.4f}$")
            st.write(rf"$\hat{{\beta_1}} = {beta_pred[1]:.4f}$")
            st.write(polynomial_to_latex(beta_pred, hat=True))
            
    else:
        st.write("#### Valores estimados")
        st.write(rf"$\hat{{\beta_0}} = {beta_pred[0]:.4f}$")
        st.write(rf"$\hat{{\beta_1}} = {beta_pred[1]:.4f}$")
        st.write(polynomial_to_latex(beta_pred, hat=True))

    fig, ax = plt.subplots()
    ax.scatter(x, y, label="Dados")
    ax.plot(x, y_pred, color="red", label="Regressão")

    ax.legend()

    st.pyplot(fig)
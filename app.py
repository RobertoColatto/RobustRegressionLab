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
    x, y, beta_true, outlier_idx, synthetic_model = synthetic_data_page()
else:
    x, y = imported_data_page()
    outlier_idx = np.array([], dtype=int)


if x is not None and y is not None:

    # escolha do modelo
    if data_source == "Gerar dados sintéticos":
        model_type = synthetic_model
    else:
        st.sidebar.header("Modelo")         

        model_type = st.sidebar.selectbox(
            "Curva de regressão",
            (
                "Polinomial",
                "Exponencial",
                "Logarítmica",
                "Potência"
            )
        )

    # polinomial
    if model_type == "Polinomial":
        if data_source == "Gerar dados sintéticos":
            degree = len(beta_true) - 1
        else:
            degree = st.sidebar.slider("Grau do polinômio", 1, 10, 1)

        X = np.column_stack([x**i for i in range(degree + 1)])

        beta_pred, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

        y_pred = X @ beta_pred

    # exponencial
    elif model_type == "Exponencial":
        if np.any(y <= 0):
            st.error("Todos os valores de y devem ser positivos para regressão exponencial.")
            st.stop()

        Y = np.log(y)

        X = np.column_stack((np.ones_like(x), x))

        alpha, beta1 = np.linalg.lstsq(X, Y, rcond=None)[0]

        beta0 = np.exp(alpha)

        beta_pred = np.array([beta0, beta1])

        y_pred = beta0 * np.exp(beta1 * x)

    # logarítmica
    elif model_type == "Logarítmica":
        if np.any(x <= 0):
            st.error("Todos os valores de x devem ser positivos para regressão logarítmica.")
            st.stop()

        X = np.column_stack((np.ones_like(x), np.log(x)))

        beta_pred, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

        y_pred = X @ beta_pred

    # potência
    elif model_type == "Potência":
        if np.any(x <= 0) or np.any(y <= 0):
            st.error("Todos os valores de x e y devem ser positivos para regressão potência.")
            st.stop()

        Y = np.log(y)

        X = np.column_stack((np.ones_like(x), np.log(x)))

        alpha, beta1 = np.linalg.lstsq(X, Y, rcond=None)[0]

        beta0 = np.exp(alpha)

        beta_pred = np.array([beta0, beta1])

        y_pred = beta0 * x**beta1

    # exibir resultados
    st.write("### Resultados do modelo")

    if data_source == "Gerar dados sintéticos":
        col1, col2 = st.columns(2)

        with col1:

            st.write("#### Valores reais")

            for i, b in enumerate(beta_true):
                st.write(rf"$\beta_{i} = {b:.4f}$")

            if model_type == "Polinomial":
                st.write(polynomial_to_latex(beta_true))

        with col2:
            st.write("#### Valores estimados")

            for i, b in enumerate(beta_pred):
                st.write(rf"$\hat{{\beta_{i}}} = {b:.4f}$")

            if model_type == "Polinomial":
                st.write(polynomial_to_latex(beta_pred, hat=True))

    else:
        st.write("#### Valores estimados")

        for i, b in enumerate(beta_pred):
            st.write(rf"$\hat{{\beta_{i}}} = {b:.4f}$")

        if model_type == "Polinomial":
            st.write(polynomial_to_latex(beta_pred, hat=True))

    x_sorted = x[np.argsort(x)]
    y_pred_sorted = y_pred[np.argsort(x)]

    fig, ax = plt.subplots()
    ax.scatter(x, y, label="Dados")
    ax.plot(x_sorted, y_pred_sorted, color="green", linewidth=3, label="Regressão")
    ax.legend()
    st.pyplot(fig)
import streamlit as st
import numpy as np


def synthetic_data_page():
    st.header("Dados sintéticos")

    beta0 = st.sidebar.slider("β₀ (Intercepto)", -10.0, 10.0, 1.0)
    beta1 = st.sidebar.slider("β₁ (Coeficiente angular)", -10.0, 10.0, 2.0)
    beta_true = np.array([beta0, beta1])

    n_points = st.sidebar.slider("Número de pontos", 10, 200, 50)
    noise = st.sidebar.slider("Desvio padrão do ruído", 0.0, 10.0, 1.0)

    x = np.linspace(0, 10, n_points)
    X = np.column_stack((np.ones_like(x), x))

    y_true = X @ beta_true
    error = np.random.normal(0, noise, n_points)
    y = y_true + error

    return x, y, beta_true
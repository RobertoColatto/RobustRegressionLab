import streamlit as st
import numpy as np


def synthetic_data_source():
    st.header("Dados sintéticos")


    # geração dos dados
    st.sidebar.header("Geração dos dados")

    n_points = st.sidebar.slider("Número de pontos", 10, 200, 50)
    noise = st.sidebar.slider("Desvio padrão do ruído", 0.0, 10.0, 1.0)
    x_min, x_max = st.sidebar.slider("Intervalo de x", -100.0, 100.0, (0.0, 10.0))


    # modelo
    st.sidebar.header("Modelo")

    model_type = st.sidebar.selectbox(
        "Formato dos dados e curva de regressão",
        (
            "Polinomial",
            "Exponencial",
            "Logarítmica",
            "Potência"
        )
    )

    if model_type == "Polinomial":
        degree = st.sidebar.slider("Grau do polinômio", 1, 10, 1)


    # coeficientes
    st.sidebar.header("Coeficientes")

    # polinomial
    if model_type == "Polinomial":
        beta_true = np.array([st.sidebar.slider(f"β{i}", -10.0, 10.0, 1.0 if i == 0 else 0.0) for i in range(degree + 1)])

        x = np.linspace(x_min, x_max, n_points)
        X = np.column_stack([x**i for i in range(degree + 1)])
        y_true = X @ beta_true

    # exponencial
    elif model_type == "Exponencial":
        beta0 = st.sidebar.slider("β₀", 0.1, 10.0, 1.0)
        beta1 = st.sidebar.slider("β₁", -5.0, 5.0, 0.3)

        beta_true = np.array([beta0, beta1])
        x = np.linspace(x_min, x_max, n_points)
        y_true = beta0 * np.exp(beta1 * x)

    # logarítmica
    elif model_type == "Logarítmica":
        beta0 = st.sidebar.slider("β₀", -10.0, 10.0, 1.0)
        beta1 = st.sidebar.slider("β₁", -10.0, 10.0, 2.0)

        beta_true = np.array([beta0, beta1])

        if x_max <= 0:
            st.error("Para dados logarítmicos, o intervalo de x deve conter valores positivos.")
            st.stop()

        x = np.linspace(max(x_min, 0.1), x_max, n_points)
        y_true = beta0 + beta1 * np.log(x)

    # potência
    else:
        beta0 = st.sidebar.slider("β₀", 0.1, 10.0, 1.0)
        beta1 = st.sidebar.slider("β₁", -5.0, 5.0, 2.0)

        beta_true = np.array([beta0, beta1])

        if x_max <= 0:
            st.error("Para dados de potência, o intervalo de x deve conter valores positivos.")
            st.stop()

        x = np.linspace(max(x_min, 0.1), x_max, n_points)
        y_true = beta0 * x**beta1


    y = y_true + np.random.normal(0, noise, n_points)


    # outliers
    st.sidebar.header("Outliers")

    use_outliers = st.sidebar.checkbox("Adicionar outliers", value=False)
    outlier_idx = np.array([], dtype=int)

    if use_outliers:
        n_outliers = st.sidebar.slider("Número de outliers", 1, min(20, n_points), 5)
        outlier_std = st.sidebar.slider("Intensidade dos outliers", 1.0, 100.0, 20.0)

        outlier_idx = np.random.choice(n_points, n_outliers, replace=False)
        y[outlier_idx] += np.random.normal(0, outlier_std, n_outliers)


    return x, y, beta_true, outlier_idx, model_type
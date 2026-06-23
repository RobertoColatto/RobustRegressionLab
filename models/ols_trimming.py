import streamlit as st
import numpy as np


def ols_trimming(X, y, beta=0.0005, epsilon=1e-6, max_iter=1000):
    n, d = X.shape

    v = np.ones(n)
    coeffs = np.zeros(d)

    k = 0
    err = 2 * epsilon

    while err > epsilon and k < max_iter:
        k += 1

        # Atualização dos coeficientes
        V = np.diag(v)
        coeffs_new = np.linalg.solve(X.T @ V @ X, X.T @ V @ y)

        # Atualização dos pesos
        residuals = (y - X @ coeffs_new) ** 2
        v_new = v - beta * residuals
        v_new = np.clip(v_new, 0, 1)

        # Critério de parada
        err = np.linalg.norm(coeffs_new - coeffs) + np.linalg.norm(v_new - v)

        coeffs, v = coeffs_new, v_new

    return coeffs, v


def polynomial_fit(data_source, x, y, beta_true):
    if data_source == "Gerar dados sintéticos":
        degree = len(beta_true) - 1
    else:
        degree = st.sidebar.slider("Grau do polinômio", 1, 10, 1)

    X = np.column_stack([x**i for i in range(degree + 1)])

    beta_pred, weights = ols_trimming(X, y)

    y_pred = X @ beta_pred

    return y_pred, beta_pred, weights


def exponential_fit(x, y):
    if np.any(y <= 0):
        st.error("Todos os valores de y devem ser positivos para regressão exponencial.")
        st.stop()

    Y = np.log(y)

    X = np.column_stack((np.ones_like(x), x))

    coeffs, weights = ols_trimming(X, Y)

    alpha, beta1 = coeffs
    beta0 = np.exp(alpha)

    beta_pred = np.array([beta0, beta1])

    y_pred = beta0 * np.exp(beta1 * x)

    return y_pred, beta_pred, weights


def logarithmic_fit(x, y):
    if np.any(x <= 0):
        st.error("Todos os valores de x devem ser positivos para regressão logarítmica.")
        st.stop()

    X = np.column_stack((np.ones_like(x), np.log(x)))

    beta_pred, weights = ols_trimming(X, y)

    y_pred = X @ beta_pred

    return y_pred, beta_pred, weights


def power_fit(x, y):
    if np.any(x <= 0) or np.any(y <= 0):
        st.error("Todos os valores de x e y devem ser positivos para regressão potência.")
        st.stop()

    Y = np.log(y)

    X = np.column_stack((np.ones_like(x), np.log(x)))

    coeffs, weights = ols_trimming(X, Y)

    alpha, beta1 = coeffs
    beta0 = np.exp(alpha)

    beta_pred = np.array([beta0, beta1])

    y_pred = beta0 * x**beta1

    return y_pred, beta_pred, weights
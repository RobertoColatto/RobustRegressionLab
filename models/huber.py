import streamlit as st
import numpy as np

#Implementação da Regressão de Huber

def huber_regression(X, y, epsilon=1.35, max_iter=100, tol=1e-6):
    w = np.zeros(X.shape[1])

    for i in range(max_iter):
        y_pred = X @ w
        r = y - y_pred

        weights = np.where(np.abs(r) <= epsilon, 1, epsilon / np.abs(r))
        W = np.diag(weights)

        w_new = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ y)

        if np.linalg.norm(w_new - w) < tol:
            break

        w = w_new

    return w

def polynomial_fit(data_source, x, y, beta_true):

    if data_source == "Gerar dados sintéticos":
        degree = len(beta_true) - 1

    else:
        degree = st.sidebar.slider("Grau do polinômio", 1, 10, 1)

    X = np.column_stack([x**i for i in range(degree + 1)])

    beta_pred = huber_regression(X, y)

    y_pred = X @ beta_pred

    return y_pred, beta_pred

def exponential_fit(x, y):

    if np.any(y <= 0):
        st.error("Todos os valores de y devem ser positivos para regressão exponencial.")
        st.stop()

    Y = np.log(y)

    X = np.column_stack((np.ones_like(x), x))

    alpha, beta1 = huber_regression(X, Y)
    beta0 = np.exp(alpha)

    beta_pred = np.array([beta0, beta1])

    y_pred = beta0 * np.exp(beta1 * x)

    return y_pred, beta_pred


def logarithmic_fit(x, y):

    if np.any(x <= 0):
        st.error("Todos os valores de x devem ser positivos para regressão logarítmica.")
        st.stop()

    X = np.column_stack((np.ones_like(x), np.log(x)))

    beta_pred = huber_regression(X, y)

    y_pred = X @ beta_pred

    return y_pred, beta_pred


def power_fit(x, y):

    if np.any(x <= 0) or np.any(y <= 0):
        st.error("Todos os valores de x e y devem ser positivos para regressão potência.")
        st.stop()

    Y = np.log(y)

    X = np.column_stack((np.ones_like(x), np.log(x)))

    alpha, beta1 = huber_regression(X, Y)
    beta0 = np.exp(alpha)

    beta_pred = np.array([beta0, beta1])

    y_pred = beta0 * x**beta1

    return y_pred, beta_pred
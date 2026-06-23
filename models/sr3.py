import streamlit as st
import numpy as np


#Implementação do algortimo SR3 básico
def prox_l1(v, Lambda):
    return np.sign(v) * np.maximum(np.abs(v) - Lambda, 0.0)

def sr3(X, X_hat, Lambda=0.1, nu=1.0, epsilon=1e-6, max_iter=1000):
    n, d = X.shape

    k = 0
    err = 2 * epsilon
    W = np.zeros(d)

    while err > epsilon and k < max_iter:
        k += 1

        #Atualizando Xi
        A = (X.T @ X + (1/nu) * np.eye(d))
        b = X.T @ X_hat + (1/nu) * W
        Xi = np.linalg.solve(A, b)

        #Atualizando W
        W_new = prox_l1(Xi, Lambda * nu)

        #Atualizano o erro
        err = np.linalg.norm(W_new - W) / nu

        W = W_new

    return Xi, W


def polynomial_fit(data_source, x, y, beta_true):

    if data_source == "Gerar dados sintéticos":
        degree = len(beta_true) - 1

    else:
        degree = st.sidebar.slider("Grau do polinômio", 1, 10, 1)

    X = np.column_stack([x**i for i in range(degree + 1)])

    Xi, W = sr3(X, y)

    beta_pred = Xi

    y_pred = X @ beta_pred

    return y_pred, beta_pred, W


def exponential_fit(x, y):

    if np.any(y <= 0):
        st.error("Todos os valores de y devem ser positivos para regressão exponencial.")
        st.stop()

    Y = np.log(y)

    X = np.column_stack((np.ones_like(x), x))

    Xi, W = sr3(X, Y)

    alpha, beta1 = Xi
    beta0 = np.exp(alpha)

    beta_pred = np.array([beta0, beta1])

    y_pred = beta0 * np.exp(beta1 * x)

    return y_pred, beta_pred, W


def logarithmic_fit(x, y):

    if np.any(x <= 0):
        st.error("Todos os valores de x devem ser positivos para regressão logarítmica.")
        st.stop()

    X = np.column_stack((np.ones_like(x), np.log(x)))

    Xi, W = sr3(X, y)

    beta_pred = Xi

    y_pred = X @ beta_pred

    return y_pred, beta_pred, W


def power_fit(x, y):

    if np.any(x <= 0) or np.any(y <= 0):
        st.error("Todos os valores de x e y devem ser positivos para regressão potência.")
        st.stop()

    Y = np.log(y)

    X = np.column_stack((np.ones_like(x), np.log(x)))

    Xi, W = sr3(X, Y)

    alpha, beta1 = Xi
    beta0 = np.exp(alpha)

    beta_pred = np.array([beta0, beta1])

    y_pred = beta0 * x**beta1

    return y_pred, beta_pred, W
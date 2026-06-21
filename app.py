import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils import polynomial_to_latex

st.title("Regressão por Mínimos Quadrados")


# sliders para definição de parâmetros
st.sidebar.header("Parâmetros")

beta_true0 = st.sidebar.slider("β₀ (Intercepto)", -10.0, 10.0, 1.0)
beta_true1 = st.sidebar.slider("β₁ (Coeficiente angular)", -10.0, 10.0, 2.0)
beta_true = beta_true0, beta_true1

n_points = st.sidebar.slider("Número de pontos", 10, 200, 50)

noise = st.sidebar.slider("Desvio padrão do ruído", 0.0, 10.0, 1.0)


# geração dos dados
x = np.linspace(0, 10, n_points)

X = np.column_stack((np.ones_like(x), x))

y_true = X @ beta_true # função real

error = np.random.normal(0, noise, n_points)

y_noisy = y_true + error # dados com ruído


# minímos quadrados
beta_pred, _, _, _ = np.linalg.lstsq(X, y_noisy, rcond=None)

y_pred = X @ beta_pred # curva de regressão


# visualização dos parâmetros
st.write("### Parâmetros")

col1, col2 = st.columns(2)

with col1:
    st.write("#### Valores reais")
    st.write(rf"$\beta_0 = {beta_true[0]:.4f}$")
    st.write(rf"$\beta_1 = {beta_true[1]:.4f}$")
    st.write(polynomial_to_latex(beta_true))

with col2:
    st.write("#### Valores estimados")
    st.write(rf"$\beta_0 = {beta_pred[0]:.4f}$")
    st.write(rf"$\beta_1 = {beta_pred[1]:.4f}$")
    st.write(polynomial_to_latex(beta_pred, hat=True))


# visualização do gráfico
fig, ax = plt.subplots()

ax.scatter(x, y_noisy, label="Dados observados")
ax.plot(x, y_true, label="Função real")
ax.plot(x, y_pred, label="Regressão OLS", linewidth=3)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()

st.pyplot(fig)

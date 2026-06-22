import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def synthetic_data_result_plot(beta_true, beta_pred):
    st.write("### Resultados do modelo")

    col1, col2 = st.columns(2)

    with col1:
        st.write("#### Valores reais")

        for i, b in enumerate(beta_true):
            st.write(rf"$\beta_{i} = {b:.4f}$")

    with col2:
        st.write("#### Valores estimados")
        
        for i, b in enumerate(beta_pred):
            st.write(rf"$\hat{{\beta_{i}}} = {b:.4f}$")


def imported_data_result_plot(beta_pred):
    st.write("### Resultados do modelo")

    st.write("#### Valores estimados")

    for i, b in enumerate(beta_pred):
        st.write(rf"$\hat{{\beta_{i}}} = {b:.4f}$")


def graphic_plot(x, y, y_pred):
    x_sorted = x[np.argsort(x)]
    y_pred_sorted = y_pred[np.argsort(x)]

    fig, ax = plt.subplots()
    ax.scatter(x, y, label="Dados")
    ax.plot(x_sorted, y_pred_sorted, color="green", linewidth=3, label="Regressão")
    ax.legend()
    st.pyplot(fig)

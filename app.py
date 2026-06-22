import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from data_sources.synthetic_data import synthetic_data_source
from data_sources.imported_data import imported_data_source
from models.ols import *
from plots.result_plot import *


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


# gerar/importar dados
if data_source == "Gerar dados sintéticos":
    x, y, beta_true, outlier_idx, synthetic_model = synthetic_data_source()

else:
    x, y = imported_data_source()
    beta_true = None
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


    # ols_fit
    if model_type == "Polinomial":
        y_pred, beta_pred = polynomial_fit(data_source, x, y, beta_true)

    elif model_type == "Exponencial":
        y_pred, beta_pred = exponential_fit(x, y)

    elif model_type == "Logarítmica":
        y_pred, beta_pred = logarithmic_fit(x, y)

    elif model_type == "Potência":
        y_pred, beta_pred = power_fit(x, y)


    # exibir resultados
    if data_source == "Gerar dados sintéticos":
        synthetic_data_result_plot(beta_true, beta_pred)

    else:
        imported_data_result_plot(beta_pred)

    graphic_plot(x, y, y_pred)
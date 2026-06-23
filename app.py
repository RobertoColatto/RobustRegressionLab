import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from data_sources.synthetic_data import synthetic_data_source
from data_sources.imported_data import imported_data_source

import models.ols as ols
import models.sr3 as sr3
import models.huber as huber
import models.ols_trimming as ols_trimming
import models.sr3_trimming as sr3_trimming

from plots.result_plot import *

if "regression_method" not in st.session_state:
    st.session_state.regression_method = "OLS"

titles = {
    "OLS": "Regressão por Mínimos Quadrados",
    "SR3": "Regressão Regularizada Relaxada Esparsa (SR3)",
    "Huber": "Regressão de Huber",
    "OLS Trimming": "Regressão por Mínimos Quadrados com Trimming",
    "SR3 Trimming": "Regressão Regularizada Relaxada Esparsa (SR3) com Trimming"
}

st.title(titles[st.session_state.regression_method])


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

    regression_method = st.session_state.regression_method


    # ols_fit
    if model_type == "Polinomial":
        if regression_method == "OLS":
            y_pred, beta_pred = ols.polynomial_fit(data_source, x, y, beta_true)
        elif regression_method == "SR3":
            y_pred, beta_pred, _ = sr3.polynomial_fit(data_source, x, y, beta_true)
        elif regression_method == "Huber":
            y_pred, beta_pred = huber.polynomial_fit(data_source, x, y, beta_true)
        elif regression_method == "OLS Trimming":
            y_pred, beta_pred, _ = ols_trimming.polynomial_fit(data_source, x, y, beta_true)
        elif regression_method == "SR3 Trimming":
            y_pred, beta_pred, _, _ = sr3_trimming.polynomial_fit(data_source, x, y, beta_true)

    elif model_type == "Exponencial":
        if regression_method == "OLS":
            y_pred, beta_pred = ols.exponential_fit(x, y)
        elif regression_method == "SR3":
            y_pred, beta_pred, _ = sr3.exponential_fit(x, y)
        elif regression_method == "Huber":
            y_pred, beta_pred = huber.exponential_fit(x, y)
        elif regression_method == "OLS Trimming":
            y_pred, beta_pred, _ = ols_trimming.exponential_fit(x, y)
        elif regression_method == "SR3 Trimming":
            y_pred, beta_pred, _, _ = sr3_trimming.exponential_fit(x, y)

    elif model_type == "Logarítmica":
        if regression_method == "OLS":
            y_pred, beta_pred = ols.logarithmic_fit(x, y)
        elif regression_method == "SR3":
            y_pred, beta_pred, _ = sr3.logarithmic_fit(x, y)
        elif regression_method == "Huber":
            y_pred, beta_pred = huber.logarithmic_fit(x, y)
        elif regression_method == "OLS Trimming":
            y_pred, beta_pred, _ = ols_trimming.logarithmic_fit(x, y)
        elif regression_method == "SR3 Trimming":
            y_pred, beta_pred, _, _ = sr3_trimming.logarithmic_fit(x, y)

    elif model_type == "Potência":
        if regression_method == "OLS":
            y_pred, beta_pred = ols.power_fit(x, y)
        elif regression_method == "SR3":
            y_pred, beta_pred, _ = sr3.power_fit(x, y)
        elif regression_method == "Huber":
            y_pred, beta_pred = huber.power_fit(x, y)
        elif regression_method == "OLS Trimming":
            y_pred, beta_pred, _ = ols_trimming.power_fit(x, y)
        elif regression_method == "SR3 Trimming":
            y_pred, beta_pred, _, _ = sr3_trimming.power_fit(x, y)


    # exibir resultados
    if data_source == "Gerar dados sintéticos":
        synthetic_data_result_plot(beta_true, beta_pred)

    else:
        imported_data_result_plot(beta_pred)

    graphic_plot(x, y, y_pred)


st.sidebar.header("Método de regressão")

st.sidebar.selectbox(
    "Estimador",
    (
        "OLS",
        "SR3",
        "Huber",
        "OLS Trimming",
        "SR3 Trimming"
    ),
    key="regression_method"
)
# Credit-Risk-Intelligence
Project Overview

A machine learning project to estimate credit default probability and explain the factors influencing individual risk predictions.

The project developed an end-to-end credit risk modeling workflow using the UCI Default of Credit Card Clients dataset. A tuned Gradient Boosting model was selected, combining feature engineering, model evaluation, and SHAP explainability to provide insights into credit risk patterns.

The goal is not to automate credit approval decisions, but to demonstrate how predictive modeling can support risk analysis and improve model interpretability.

Key Results

Model: Gradient Boosting Classifier

ROC-AUC: 0.788

Average Precision: 0.562

KS Statistic: 0.437

Cross-validation: 5-fold stratified validation

These metrics are pooled out-of-fold estimates from cross-validation.

Methodology

Exploratory Data Analysis: Examined customer characteristics, repayment behavior, and default patterns.

Feature Engineering: Created additional indicators related to payment delays, credit utilization, payment behavior, and billing trends.

Model Development: Compared baseline models and tuned the selected Gradient Boosting model using randomized search and cross-validation.

Model Explainability: Applied SHAP to analyze global feature importance, feature impact patterns, and individual predictions.

Model Explainability

The SHAP analysis highlighted payment behavior and delinquency-related features as major contributors to model predictions.

Global Feature Importance



SHAP Summary



Individual Prediction Explanation



Technologies

Python · Pandas · NumPy · Scikit-learn · SHAP · Matplotlib · Seaborn · Joblib · Jupyter Notebook

Repository Structure

notebooks/: Exploratory analysis, modeling, feature engineering, and explainability.

src/: Reusable feature engineering components.

data/: Dataset and processed modeling data.

models/: Serialized machine learning pipeline.

images/: Visualizations used in this documentation.

Dataset and Limitations

Source: UCI – Default of Credit Card Clients

The dataset contains historical credit card client information from Taiwan. Its results should not be assumed to generalize to current customers or other markets.

The test set was used during model comparison and should not be considered an independent final evaluation. SHAP values describe model behavior and associations, not causal relationships.

This project is intended for educational and portfolio purposes, not for real-world credit approval decisions.

Credit Risk Intelligence

Descripción del Proyecto

Proyecto de machine learning orientado a estimar la probabilidad de incumplimiento crediticio y explicar los factores que influyen en las predicciones individuales de riesgo.

Se desarrolló un flujo de trabajo integral de modelamiento de riesgo crediticio utilizando el dataset UCI Default of Credit Card Clients. Se seleccionó un modelo Gradient Boosting optimizado, integrando ingeniería de características, evaluación de modelos y explicabilidad mediante SHAP.

El objetivo no es automatizar decisiones de aprobación crediticia, sino demostrar cómo los modelos predictivos pueden apoyar el análisis de riesgo y mejorar la interpretabilidad.

Resultados Principales

Modelo: Gradient Boosting Classifier

ROC-AUC: 0.788

Average Precision: 0.562

Estadístico KS: 0.437

Validación cruzada: 5-fold estratificada

Estas métricas corresponden a estimaciones agrupadas out-of-fold obtenidas mediante validación cruzada.

Metodología

Análisis Exploratorio: Evaluación de características de clientes, comportamiento de pago y patrones de incumplimiento.

Ingeniería de Características: Creación de indicadores relacionados con retrasos, utilización del crédito, comportamiento de pago y tendencias de facturación.

Desarrollo del Modelo: Comparación de modelos base y optimización de Gradient Boosting mediante búsqueda aleatoria y validación cruzada.

Explicabilidad: Aplicación de SHAP para analizar la importancia global de las variables, sus patrones de impacto y predicciones individuales.

Tecnologías

Python · Pandas · NumPy · Scikit-learn · SHAP · Matplotlib · Seaborn · Joblib · Jupyter Notebook

Dataset y Limitaciones

Fuente: UCI – Default of Credit Card Clients

El dataset contiene información histórica de clientes de tarjetas de crédito de Taiwán. Los resultados no deben extrapolarse automáticamente a clientes actuales ni a otros mercados.

El conjunto de prueba fue utilizado durante la comparación de modelos, por lo que no constituye una evaluación final independiente. Los valores SHAP describen el comportamiento y las asociaciones del modelo, no relaciones causales.

Este proyecto tiene fines educativos y de portafolio, y no está diseñado para tomar decisiones reales de aprobación crediticia.

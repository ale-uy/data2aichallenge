# Simulador de Estrés de Demanda Energética

Este proyecto es una aplicación web interactiva construida con Streamlit que permite simular y visualizar el impacto de diferentes escenarios climáticos en la demanda de energía eléctrica.

[![Streamlit App](https://data2aichallenge-aleuy.streamlit.app/)

## Descripción

La herramienta utiliza un conjunto de modelos de Machine Learning preentrenados para:
1.  **Pronosticar la demanda** de energía para una fecha futura.
2.  **Simular escenarios de estrés** manipulando variables como la temperatura, olas de calor/frío y el fenómeno ENSO.
3.  **Visualizar los riesgos**, comparando la predicción con umbrales de estrés alto (percentil 95) y extremo (percentil 99).
4.  **Realizar backtesting** comparando las predicciones del modelo con datos históricos reales.

## Archivos del Repositorio

Este repositorio contiene únicamente los archivos necesarios para desplegar y ejecutar la aplicación Streamlit.

*   **Script Principal:**
    *   `app.py`: El código de la aplicación web.

*   **Datos Requeridos (`.csv`):**
    *   `demanda_con_clima.csv`: Datos históricos de demanda y clima ya procesados.
    *   `pronostico_temperaturas_futuras.csv`: Predicciones de temperatura generadas por un modelo Prophet.
    *   `pronostico_demanda_final.csv`: Pronóstico base de la demanda a 18 meses.

*   **Modelos Preentrenados (`.pkl`):**
    *   `trend_model_optimized.pkl` / `..._noche.pkl`: Modelos de tendencia lineal.
    *   `xgb_optimized_residual_model.pkl` / `..._noche.pkl`: Modelos XGBoost para la predicción central de residuos.
    *   `xgb_dia_q95.pkl` / `xgb_noche_q95.pkl`: Modelos para el escenario de "Alto Estrés".
    *   `xgb_dia_q99.pkl` / `xgb_noche_q99.pkl`: Modelos para el escenario de "Estrés Extremo".
    
*   **Dependencias:**
    *   `requirements.txt`: Lista de las librerías de Python necesarias.

## ¿Cómo Desplegar en Streamlit Community Cloud?

1.  **Crea un Repositorio en GitHub:** Crea un nuevo repositorio público.
2.  **Sube los Archivos:** Sube todos los archivos listados en la sección anterior a tu repositorio.
3.  **Regístrate en Streamlit:** Ve a [share.streamlit.io](https://share.streamlit.io) y regístrate usando tu cuenta de GitHub.
4.  **Crea una Nueva App:**
    *   Haz clic en "New app".
    *   Selecciona tu repositorio y la rama principal (main).
    *   Asegúrate de que el "Main file path" apunte a `6_streamlit_app.py`.
    *   Haz clic en "Deploy!".
5.  **¡Listo!** Streamlit se encargará de instalar las dependencias de tu `requirements.txt` y de ejecutar la aplicación.

## ¿Cómo Ejecutar Localmente?

1.  **Clona el Repositorio:**
    ```bash
    git clone <URL_de_tu_repositorio>
    cd <nombre_del_repositorio>
    ```
2.  **Crea un Entorno Virtual (Recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```
3.  **Instala las Dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Ejecuta la Aplicación:**
    ```bash
    streamlit run app.py
    ```

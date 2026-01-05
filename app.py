import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import holidays
import plotly.graph_objects as go
from datetime import date

# --- Configuración de la Página ---
st.set_page_config(page_title="Simulador de Demanda Energética", page_icon="⚡️", layout="wide")

# --- 1. Carga de Artefactos ---
@st.cache_resource
def load_artifacts():
    """Carga todos los modelos y datos necesarios una sola vez."""
    print("Cargando todos los modelos y datos...")
    artifacts = {}
    try:
        artifacts['df_historico'] = pd.read_csv('demanda_con_clima.csv', parse_dates=['fecha'])
        artifacts['df_pronostico_temps'] = pd.read_csv('pronostico_temperaturas_futuras.csv', parse_dates=['fecha'])
        artifacts['df_pronostico_final'] = pd.read_csv('pronostico_demanda_final.csv', parse_dates=['fecha'])
        
        # Modelos DÍA (Central + Estrés)
        artifacts['trend_model_dia'] = joblib.load('trend_model_optimized.pkl')
        artifacts['xgb_residual_dia'] = joblib.load('xgb_optimized_residual_model.pkl')
        artifacts['xgb_dia_q95'] = joblib.load('xgb_dia_q95.pkl') # <-- CAMBIO
        artifacts['xgb_dia_q99'] = joblib.load('xgb_dia_q99.pkl')
        
        # Modelos NOCHE (Central + Estrés)
        artifacts['trend_model_noche'] = joblib.load('trend_model_optimized_noche.pkl')
        artifacts['xgb_residual_noche'] = joblib.load('xgb_optimized_residual_model_noche.pkl')
        artifacts['xgb_noche_q95'] = joblib.load('xgb_noche_q95.pkl') # <-- CAMBIO
        artifacts['xgb_noche_q99'] = joblib.load('xgb_noche_q99.pkl')

        print("Carga de artefactos completada.")
        return artifacts
    except FileNotFoundError as e:
        st.error(f"Error al cargar un archivo esencial: {e.filename}. Asegúrate de haber ejecutado todos los scripts de entrenamiento.")
        return None

# --- 2. Funciones de Lógica ---
# (La función crear_features_para_un_dia no necesita cambios)
@st.cache_data
def crear_features_para_un_dia(_artifacts, target_date, t_max, t_min, enso_status, 
                               is_heatwave_override, is_coldsnap_override, 
                               use_avg_max_3d_override, avg_temp_max_3d_override,
                               use_avg_min_3d_override, avg_temp_min_3d_override,
                               use_avg_max_7d_override, avg_temp_max_7d_override,
                               use_avg_min_7d_override, avg_temp_min_7d_override):
    """
    Crea el vector de características completo para una única fecha futura,
    con la opción de sobrescribir características de escenario.
    """
    target_date = pd.to_datetime(target_date)
    df_historico = _artifacts['df_historico']
    
    columnas_base = ['fecha', 'T_Max_Index', 'T_Min_Index']
    hist_buffer = df_historico[df_historico['fecha'] < target_date][columnas_base].tail(30)
    future_df = pd.DataFrame({'fecha': [target_date], 'T_Max_Index': [t_max], 'T_Min_Index': [t_min]})
    df_combinado = pd.concat([hist_buffer, future_df], ignore_index=True)
    df_combinado['fecha'] = pd.to_datetime(df_combinado['fecha'])
    
    mes = df_combinado['fecha'].dt.month
    df_combinado['mes_sin'] = np.sin(2 * np.pi * mes / 12)
    df_combinado['mes_cos'] = np.cos(2 * np.pi * mes / 12)
    df_combinado['dia_semana'] = df_combinado['fecha'].dt.dayofweek
    df_combinado['dia_del_anio'] = df_combinado['fecha'].dt.dayofyear
    df_combinado['semana_del_anio'] = df_combinado['fecha'].dt.isocalendar().week.astype(int)
    uy_holidays = holidays.Uruguay(years=target_date.year)
    feriados_predecibles = {"inamovibles": ["New Year's Day", "Workers' Day", "Constitution Day", "Independence Day", "Day of the Family"], "turismo": ["Tourism Week"], "carnaval": ["Carnival"]}
    def es_feriado_predecible(fecha):
        holiday_name = uy_holidays.get(fecha)
        if holiday_name is None: return 0
        for keywords in feriados_predecibles.values():
            if any(keyword in holiday_name for keyword in keywords): return 1
        return 0
    df_combinado['es_feriado'] = df_combinado['fecha'].apply(es_feriado_predecible).astype(int)
    df_combinado['es_finde'] = (df_combinado['dia_semana'] >= 5).astype(int)
    for i in range(1, 4):
        df_combinado[f'T_Max_Index_Lag{i}'] = df_combinado['T_Max_Index'].shift(i)
        df_combinado[f'T_Min_Index_Lag{i}'] = df_combinado['T_Min_Index'].shift(i)
    
    df_combinado['T_Max_3d_avg'] = df_combinado['T_Max_Index'].rolling(window=3).mean()
    df_combinado['T_Min_3d_avg'] = df_combinado['T_Min_Index'].rolling(window=3).mean()
    df_combinado['T_Max_7d_avg'] = df_combinado['T_Max_Index'].rolling(window=7).mean()
    df_combinado['T_Min_7d_avg'] = df_combinado['T_Min_Index'].rolling(window=7).mean()
    
    if use_avg_max_3d_override: df_combinado.loc[df_combinado.index[-1], 'T_Max_3d_avg'] = avg_temp_max_3d_override
    if use_avg_min_3d_override: df_combinado.loc[df_combinado.index[-1], 'T_Min_3d_avg'] = avg_temp_min_3d_override
    if use_avg_max_7d_override: df_combinado.loc[df_combinado.index[-1], 'T_Max_7d_avg'] = avg_temp_max_7d_override
    if use_avg_min_7d_override: df_combinado.loc[df_combinado.index[-1], 'T_Min_7d_avg'] = avg_temp_min_7d_override

    cond_calor = (df_combinado['T_Min_Index'] >= 21) & (df_combinado['T_Max_Index'] >= 34)
    df_combinado['ola_de_calor'] = (cond_calor.astype(int).rolling(window=3).sum() >= 3).astype(int)
    if is_heatwave_override: df_combinado.loc[df_combinado.index[-1], 'ola_de_calor'] = 1

    cond_frio = (df_combinado['T_Min_Index'] < 3) & (df_combinado['T_Max_Index'] < 12)
    df_combinado['ola_de_frio'] = (cond_frio.astype(int).rolling(window=3).sum() >= 3).astype(int)
    if is_coldsnap_override: df_combinado.loc[df_combinado.index[-1], 'ola_de_frio'] = 1
    
    df_combinado['Tmax_x_mes_sin'] = df_combinado['T_Max_Index'] * df_combinado['mes_sin']
    df_combinado['Tmax_x_mes_cos'] = df_combinado['T_Max_Index'] * df_combinado['mes_cos']
    df_combinado['Tmin_x_mes_sin'] = df_combinado['T_Min_Index'] * df_combinado['mes_sin']
    df_combinado['Tmin_x_mes_cos'] = df_combinado['T_Min_Index'] * df_combinado['mes_cos']
    df_combinado['enso_El Niño'] = 1 if enso_status == 'El Niño' else 0
    df_combinado['enso_La Niña'] = 1 if enso_status == 'La Niña' else 0
    
    return df_combinado.tail(1)

# --- 3. Construcción de la Interfaz ---
st.title("Simulador de Estrés Energético ⚡️")
st.markdown("Herramienta de análisis dual: **pronóstico base** y **simulación de escenarios de riesgo**.")

artifacts = load_artifacts()

if artifacts:
    st.sidebar.header("Modo de Operación")
    modo_operacion = st.sidebar.radio("Elige un modo:", ('Pronóstico Automático', 'Simulación Manual'))
    fecha_input = st.sidebar.date_input("1. Selecciona una fecha", value=pd.to_datetime("2026-01-20"))

    # (El resto de la lógica de la sidebar no necesita cambios)
    is_heatwave_override, is_coldsnap_override = False, False
    use_avg_max_3d, avg_temp_max_3d = False, 0
    use_avg_min_3d, avg_temp_min_3d = False, 0
    use_avg_max_7d, avg_temp_max_7d = False, 0
    use_avg_min_7d, avg_temp_min_7d = False, 0
    
    if modo_operacion == 'Pronóstico Automático':
        df_temps = artifacts['df_pronostico_temps']
        fecha_dt = pd.to_datetime(fecha_input)
        temp_data = df_temps[df_temps['fecha'] == fecha_dt]
        if not temp_data.empty:
            t_max, t_min = temp_data['T_Max_Index'].iloc[0], temp_data['T_Min_Index'].iloc[0]
            st.sidebar.info(f"Escenario base (Prophet):\nMáx: {t_max:.1f}°C, Mín: {t_min:.1f}°C")
        else:
            st.sidebar.warning("Sin pronóstico de Prophet. Usando valores por defecto.")
            t_max, t_min = 30, 20
        
        if fecha_dt < pd.to_datetime("2025-06-01"): enso_input = 'La Niña'
        else: enso_input = 'Neutral'
        st.sidebar.info(f"ENSO Automático: {enso_input}")
    
    else: # Modo Simulación Manual
        with st.sidebar.expander("2. Controles de Simulación Manual", expanded=True):
            t_max = st.slider("Temperatura Máxima (°C)", -5, 45, 35)
            t_min = st.slider("Temperatura Mínima (°C)", -10, 30, 22)
            enso_input = st.radio("Condición ENSO", ['Neutral', 'El Niño', 'La Niña'], index=1)
            st.markdown("---")
            is_heatwave_override = st.checkbox("Forzar 'Ola de Calor'")
            is_coldsnap_override = st.checkbox("Forzar 'Ola de Frío'")
            st.markdown("---")
            use_avg_max_3d = st.checkbox("Sobrescribir Promedio T. Máx (3 días)")
            avg_temp_max_3d = st.slider("Valor Prom. T. Máx (3d)", 0, 40, 32, disabled=not use_avg_max_3d)
            use_avg_min_3d = st.checkbox("Sobrescribir Promedio T. Mín (3 días)")
            avg_temp_min_3d = st.slider("Valor Prom. T. Mín (3d)", -10, 30, 20, disabled=not use_avg_min_3d)
            st.markdown("---")
            use_avg_max_7d = st.checkbox("Sobrescribir Promedio T. Máx (7 días)")
            avg_temp_max_7d = st.slider("Valor Prom. T. Máx (7d)", 0, 40, 32, disabled=not use_avg_max_7d)
            use_avg_min_7d = st.checkbox("Sobrescribir Promedio T. Mín (7 días)")
            avg_temp_min_7d = st.slider("Valor Prom. T. Mín (7d)", -10, 30, 20, disabled=not use_avg_min_7d)

    if st.sidebar.button("Simular Escenario", type="primary", use_container_width=True):
        with st.spinner('Calculando...'):
            df_futuro_features = crear_features_para_un_dia(
                artifacts, fecha_input, t_max, t_min, enso_input,
                is_heatwave_override, is_coldsnap_override,
                use_avg_max_3d, avg_temp_max_3d, use_avg_min_3d, avg_temp_min_3d,
                use_avg_max_7d, avg_temp_max_7d, use_avg_min_7d, avg_temp_min_7d
            )
            time_index_futuro = (pd.to_datetime(fecha_input) - artifacts['df_historico']['fecha'].min()).days
            
            p_dia = artifacts['trend_model_dia'].predict([[time_index_futuro]])[0] + artifacts['xgb_residual_dia'].predict(df_futuro_features.reindex(columns=artifacts['xgb_residual_dia'].feature_names_in_, fill_value=0))[0]
            p_dia_q95 = artifacts['trend_model_dia'].predict([[time_index_futuro]])[0] + artifacts['xgb_dia_q95'].predict(df_futuro_features.reindex(columns=artifacts['xgb_dia_q95'].feature_names_in_, fill_value=0))[0]
            p_dia_q99 = artifacts['trend_model_dia'].predict([[time_index_futuro]])[0] + artifacts['xgb_dia_q99'].predict(df_futuro_features.reindex(columns=artifacts['xgb_dia_q99'].feature_names_in_, fill_value=0))[0]
            
            p_noche = artifacts['trend_model_noche'].predict([[time_index_futuro]])[0] + artifacts['xgb_residual_noche'].predict(df_futuro_features.reindex(columns=artifacts['xgb_residual_noche'].feature_names_in_, fill_value=0))[0]
            p_noche_q95 = artifacts['trend_model_noche'].predict([[time_index_futuro]])[0] + artifacts['xgb_noche_q95'].predict(df_futuro_features.reindex(columns=artifacts['xgb_noche_q95'].feature_names_in_, fill_value=0))[0]
            p_noche_q99 = artifacts['trend_model_noche'].predict([[time_index_futuro]])[0] + artifacts['xgb_noche_q99'].predict(df_futuro_features.reindex(columns=artifacts['xgb_noche_q99'].feature_names_in_, fill_value=0))[0]

            st.header("📈 Resultados del Escenario Seleccionado")
            st.write(f"**Escenario para el día:** `{fecha_input.strftime('%Y-%m-%d')}`")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Pico Mediodía")
                fig_dia = go.Figure(go.Indicator(mode="gauge+number", value=int(p_dia), title={'text': "Predicción Central"}, gauge={'axis': {'range': [p_dia*0.8, p_dia_q99*1.1], 'tickvals': [p_dia, p_dia_q95, p_dia_q99], 'ticktext': [f"<b>{p_dia:.0f}</b>", f"{p_dia_q95:.0f}", f"{p_dia_q99:.0f}"]}, 'bar': {'color': "darkorange", 'thickness': 0}, 'steps': [{'range': [p_dia, p_dia_q95], 'color': "rgba(255, 215, 0, 0.5)"}, {'range': [p_dia_q95, p_dia_q99], 'color': "rgba(255, 69, 0, 0.5)"}], 'threshold': {'line': {'color': "red", 'width': 4}, 'value': p_dia_q99}}))
                st.plotly_chart(fig_dia, use_container_width=True)
            with col2:
                st.subheader("Pico Noche")
                fig_noche = go.Figure(go.Indicator(mode="gauge+number", value=int(p_noche), title={'text': "Predicción Central"}, gauge={'axis': {'range': [p_noche*0.8, p_noche_q99*1.1], 'tickvals': [p_noche, p_noche_q95, p_noche_q99], 'ticktext': [f"<b>{p_noche:.0f}</b>", f"{p_noche_q95:.0f}", f"{p_noche_q99:.0f}"]}, 'bar': {'color': "cornflowerblue", 'thickness': 0}, 'steps': [{'range': [p_noche, p_noche_q95], 'color': "rgba(255, 215, 0, 0.5)"}, {'range': [p_noche_q95, p_noche_q99], 'color': "rgba(255, 69, 0, 0.5)"}], 'threshold': {'line': {'color': "red", 'width': 4}, 'value': p_noche_q99}}))
                st.plotly_chart(fig_noche, use_container_width=True)

            with st.expander("ℹ️ Acerca de la Calibración de los Niveles de Estrés"):
                st.markdown("""
                La **cobertura real** se calcula a partir de la validación del modelo con datos históricos no vistos durante el entrenamiento. Representa la frecuencia con la que la demanda real ha quedado por debajo de nuestros umbrales pronosticados.

                **Pico Mediodía:**
                - **Nivel de Estrés Alto (Objetivo 95%):** Cobertura Real de **88.9%**
                - **Nivel de Estrés Extremo (Objetivo 99%):** Cobertura Real de **94.3%**

                **Pico Noche:**
                - **Nivel de Estrés Alto (Objetivo 95%):** Cobertura Real de **83.8%**
                - **Nivel de Estrés Extremo (Objetivo 99%):** Cobertura Real de **95.8%**

                *Una cobertura del 89% significa que, en la práctica, se espera que la demanda real supere este umbral de estrés el 11% de las veces, en lugar del 5% teórico.*
                """)

            # --- Lógica condicional para mostrar gráficos de serie de tiempo ---
            fecha_dt_logic = pd.to_datetime(fecha_input)
            cutoff_date = pd.to_datetime("2025-12-01")

            if fecha_dt_logic >= cutoff_date:
                st.divider()
                st.header("⏳ Evolución del Pronóstico a 18 Meses")
                df_p_full = artifacts['df_pronostico_final']
                fecha_limite = df_p_full['fecha'].min() + pd.DateOffset(months=18)
                df_p = df_p_full[df_p_full['fecha'] <= fecha_limite].copy()
                
                st.subheader("Pronóstico Pico Mediodía")
                fig_ts_dia = go.Figure()
                fig_ts_dia.add_trace(go.Scatter(x=df_p['fecha'], y=df_p['demanda_dia'], line=dict(width=0), showlegend=False, hoverinfo='none'))
                fig_ts_dia.add_trace(go.Scatter(x=df_p['fecha'], y=df_p['demanda_dia_q95'], fill='tonexty', fillcolor='rgba(255, 215, 0, 0.2)', line=dict(width=0), name='Alto Estrés (95%)', hoverinfo='none'))
                fig_ts_dia.add_trace(go.Scatter(x=df_p['fecha'], y=df_p['demanda_dia_q95'], line=dict(width=0), showlegend=False, hoverinfo='none'))
                fig_ts_dia.add_trace(go.Scatter(x=df_p['fecha'], y=df_p['demanda_dia_q99'], fill='tonexty', fillcolor='rgba(255, 69, 0, 0.2)', line=dict(width=0), name='Estrés Extremo (99%)', hoverinfo='none'))
                fig_ts_dia.add_trace(go.Scatter(x=df_p['fecha'], y=df_p['demanda_dia'], name='Pronóstico Base', line=dict(color='darkorange', width=2)))
                if modo_operacion == 'Simulación Manual':
                    fig_ts_dia.add_trace(go.Scatter(x=[pd.to_datetime(fecha_input)], y=[p_dia], mode='markers', marker=dict(color='red', size=12, symbol='star'), name='Tu Simulación'))
                fig_ts_dia.update_layout(xaxis_title="Fecha", yaxis_title="MW", height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_ts_dia, use_container_width=True)

                st.subheader("Pronóstico Pico Noche")
                fig_ts_noche = go.Figure()
                fig_ts_noche.add_trace(go.Scatter(x=df_p['fecha'], y=df_p['demanda_noche'], line=dict(width=0), showlegend=False, hoverinfo='none'))
                fig_ts_noche.add_trace(go.Scatter(x=df_p['fecha'], y=df_p['demanda_noche_q95'], fill='tonexty', fillcolor='rgba(255, 215, 0, 0.2)', line=dict(width=0), name='Alto Estrés (95%)', hoverinfo='none'))
                fig_ts_noche.add_trace(go.Scatter(x=df_p['fecha'], y=df_p['demanda_noche_q95'], name='_q95_line', line=dict(width=0), showlegend=False, hoverinfo='none'))
                fig_ts_noche.add_trace(go.Scatter(x=df_p['fecha'], y=df_p['demanda_noche_q99'], fill='tonexty', fillcolor='rgba(255, 69, 0, 0.2)', line=dict(width=0), name='Estrés Extremo (99%)', hoverinfo='none'))
                fig_ts_noche.add_trace(go.Scatter(x=df_p['fecha'], y=df_p['demanda_noche'], name='Pronóstico Base', line=dict(color='cornflowerblue', width=2)))
                if modo_operacion == 'Simulación Manual':
                    fig_ts_noche.add_trace(go.Scatter(x=[pd.to_datetime(fecha_input)], y=[p_noche], mode='markers', marker=dict(color='red', size=12, symbol='star'), name='Tu Simulación'))
                fig_ts_noche.update_layout(xaxis_title="Fecha", yaxis_title="MW", height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_ts_noche, use_container_width=True)
            else:
                st.divider()
                st.info("Los gráficos de pronóstico a largo plazo solo se muestran para fechas futuras (a partir del 1 de diciembre de 2025).")

            # (La sección de Backtesting no necesita cambios)
            st.divider()
            st.header("🔬 Backtesting con Datos Históricos")
            fecha_dt_input = pd.to_datetime(fecha_input)
            artifacts['df_historico']['fecha'] = pd.to_datetime(artifacts['df_historico']['fecha'])
            historical_data = artifacts['df_historico'][artifacts['df_historico']['fecha'].dt.date == fecha_dt_input.date()]
            if not historical_data.empty:
                actual_dia, actual_noche = historical_data['pico_mediodia_mw'].iloc[0], historical_data['pico_noche_mw'].iloc[0]
                actual_t_max, actual_t_min = historical_data['T_Max_Index'].iloc[0], historical_data['T_Min_Index'].iloc[0]
                st.info(f"La fecha seleccionada tiene datos históricos. Comparando la simulación con los valores reales del día: T.Máx: {actual_t_max:.1f}°C, T.Mín: {actual_t_min:.1f}°C.")
                col1_bt, col2_bt = st.columns(2)
                with col1_bt:
                    st.subheader("Comparativa Mediodía")
                    st.metric(label="Demanda Real", value=f"{actual_dia:.0f} MW")
                    st.metric(label="Demanda Simulada", value=f"{p_dia:.0f} MW", delta=f"{p_dia - actual_dia:.0f} MW (Error)", delta_color="inverse")
                with col2_bt:
                    st.subheader("Comparativa Noche")
                    st.metric(label="Demanda Real", value=f"{actual_noche:.0f} MW")
                    st.metric(label="Demanda Simulada", value=f"{p_noche:.0f} MW", delta=f"{p_noche - actual_noche:.0f} MW (Error)", delta_color="inverse")
                st.caption("El 'Error' es la diferencia entre el valor simulado (con los parámetros que elegiste) y el valor real histórico. Un ▲ rojo indica una sobreestimación, mientras que un ▼ verde indica una subestimación.")
            else:
                st.info("La fecha seleccionada no tiene datos históricos disponibles en 'demanda_con_clima.csv' para realizar backtesting.")

    else:
        st.info("Configura un escenario en la barra lateral y haz clic en 'Simular Escenario'.")

st.markdown("---")
st.caption("Desarrollado por Luis Silvera - Proyecto de Análisis de Estrés Energético.")

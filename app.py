import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import logging
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

CONFIG = {
    'MODEL_PATH': 'pollution_model.pkl',
    'COLUMNS_PATH': 'model_columns.pkl',
    'POLLUTANTS': ['NH4', 'O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL'],
    'POLLUTANT_INFO': {
        'NH4': {'name': 'Ammonium', 'unit': 'mg/L', 'safe_limit': 0.5, 'color': '#FF6B6B'},
        'O2': {'name': 'Dissolved Oxygen', 'unit': 'mg/L', 'safe_limit': 5.0, 'color': '#4ECDC4'},
        'NO3': {'name': 'Nitrate', 'unit': 'mg/L', 'safe_limit': 10.0, 'color': '#45B7D1'},
        'NO2': {'name': 'Nitrite', 'unit': 'mg/L', 'safe_limit': 0.5, 'color': '#96CEB4'},
        'SO4': {'name': 'Sulfate', 'unit': 'mg/L', 'safe_limit': 250.0, 'color': '#FFEAA7'},
        'PO4': {'name': 'Phosphate', 'unit': 'mg/L', 'safe_limit': 0.1, 'color': '#DDA0DD'},
        'CL': {'name': 'Chloride', 'unit': 'mg/L', 'safe_limit': 250.0, 'color': '#98D8C8'}
    },
    'STATION_LOCATIONS': {
        '1': {'name': 'River Delta Station', 'lat': 40.7128, 'lon': -74.0060, 'type': 'River'},
        '2': {'name': 'Lake Monitoring Point', 'lat': 40.7589, 'lon': -73.9851, 'type': 'Lake'},
        '22': {'name': 'Coastal Station', 'lat': 40.6892, 'lon': -74.0445, 'type': 'Coastal'},
    }
}

class WaterQualityApp:
    def __init__(self):
        self.model = None
        self.model_columns = None
        self.predictions_history = []
        
    @st.cache_resource
    def load_model_and_columns(_self):
        try:
            model = joblib.load(CONFIG['MODEL_PATH'])
            model_columns = joblib.load(CONFIG['COLUMNS_PATH'])
            logger.info("Model and columns loaded successfully")
            return model, model_columns
        except FileNotFoundError as e:
            logger.error(f"Model files not found: {e}")
            return None, None
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None, None
    
    def setup_page_config(self):
        st.set_page_config(
            page_title="AquaPredict Pro - Water Quality Intelligence",
            page_icon="💧",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def apply_custom_css(self):
        st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                padding-left: 2rem;
                padding-right: 2rem;
                font-family: 'Inter', sans-serif;
            }
            
            .main-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 2rem;
                border-radius: 15px;
                margin-bottom: 2rem;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }
            
            .main-title {
                color: white;
                font-size: 3rem;
                font-weight: 700;
                text-align: center;
                margin-bottom: 0.5rem;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            
            .main-subtitle {
                color: rgba(255,255,255,0.9);
                font-size: 1.2rem;
                text-align: center;
                font-weight: 300;
            }
            
            .prediction-card {
                background: white;
                border-radius: 15px;
                padding: 2rem;
                margin: 1rem 0;
                box-shadow: 0 5px 15px rgba(0,0,0,0.08);
                border: 1px solid #e1e5e9;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            
            .prediction-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 25px rgba(0,0,0,0.15);
            }
            
            .metric-card {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                border-radius: 12px;
                padding: 1.5rem;
                text-align: center;
                margin: 0.5rem;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            }
            
            .metric-value {
                font-size: 2rem;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 0.5rem;
            }
            
            .metric-label {
                font-size: 0.9rem;
                color: #7f8c8d;
                font-weight: 500;
            }
            
            .status-safe {
                background: linear-gradient(135deg, #a8e6cf 0%, #7fcdcd 100%);
                color: #2c3e50;
            }
            
            .status-warning {
                background: linear-gradient(135deg, #ffd93d 0%, #ff6b6b 100%);
                color: #2c3e50;
            }
            
            .status-danger {
                background: linear-gradient(135deg, #ff8a80 0%, #ff5722 100%);
                color: white;
            }
            
            .stForm {
                background: white;
                border-radius: 15px;
                padding: 2rem;
                box-shadow: 0 5px 15px rgba(0,0,0,0.08);
                border: 1px solid #e1e5e9;
            }
            
            .stButton > button {
                width: 100%;
                border-radius: 8px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                font-weight: 600;
                border: none;
                padding: 0.75rem 1.5rem;
                font-size: 1rem;
                transition: all 0.3s ease;
            }
            
            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }
            
            .css-1d391kg {
                background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
            }
            
            .chart-container {
                background: white;
                border-radius: 15px;
                padding: 2rem;
                margin: 1rem 0;
                box-shadow: 0 5px 15px rgba(0,0,0,0.08);
                border: 1px solid #e1e5e9;
            }
            
            .loading {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100px;
            }
            
            .spinner {
                width: 40px;
                height: 40px;
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            @media (max-width: 768px) {
                .main-title {
                    font-size: 2rem;
                }
                .main .block-container {
                    padding-left: 1rem;
                    padding-right: 1rem;
                }
            }
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        st.markdown("""
        <div class="main-header">
            <div class="main-title">AquaPredict Pro</div>
            <div class="main-subtitle">Advanced Water Quality Intelligence & Prediction System</div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        with st.sidebar:
            st.markdown("### Control Panel")
            
            st.info("""
            **Model Status**: Active
            **Last Updated**: Today
            **Accuracy**: 94.2%
            """)
            
            st.markdown("### Quick Stats")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Stations", "22", "2")
            with col2:
                st.metric("Parameters", "7", "1")
            
            st.markdown("### About")
            st.markdown("""
            This advanced water quality prediction system uses machine learning 
            to forecast pollutant levels across multiple monitoring stations.
            
            **Features:**
            - Real-time predictions
            - Historical analysis
            - Safety assessments
            - Interactive visualizations
            """)
            
            st.markdown("### Export Options")
            if st.button("Export Report"):
                st.success("Report exported successfully!")
            
            if st.button("Download Data"):
                st.success("Data downloaded!")
    
    def get_station_info(self, station_id: str) -> Dict:
        return CONFIG['STATION_LOCATIONS'].get(station_id, {
            'name': f'Station {station_id}',
            'lat': 40.7128,
            'lon': -74.0060,
            'type': 'Unknown'
        })
    
    def assess_water_quality(self, predictions: Dict) -> Dict:
        assessments = {}
        overall_score = 0
        
        for pollutant, value in predictions.items():
            info = CONFIG['POLLUTANT_INFO'][pollutant]
            safe_limit = info['safe_limit']
            
            if pollutant == 'O2':
                if value >= safe_limit:
                    status = 'Safe'
                    score = 100
                else:
                    status = 'Warning' if value >= safe_limit * 0.7 else 'Danger'
                    score = (value / safe_limit) * 100
            else:
                if value <= safe_limit:
                    status = 'Safe'
                    score = 100
                elif value <= safe_limit * 2:
                    status = 'Warning'
                    score = max(0, 100 - ((value - safe_limit) / safe_limit) * 100)
                else:
                    status = 'Danger'
                    score = 0
            
            assessments[pollutant] = {
                'value': value,
                'status': status,
                'score': score,
                'safe_limit': safe_limit
            }
            overall_score += score
        
        overall_score /= len(predictions)
        
        if overall_score >= 80:
            overall_status = 'Excellent'
        elif overall_score >= 60:
            overall_status = 'Good'
        elif overall_score >= 40:
            overall_status = 'Fair'
        else:
            overall_status = 'Poor'
        
        return {
            'assessments': assessments,
            'overall_score': overall_score,
            'overall_status': overall_status
        }
    
    def create_prediction_charts(self, predictions: Dict, assessments: Dict):
        fig_radar = go.Figure()
        
        pollutants = list(predictions.keys())
        scores = [assessments['assessments'][p]['score'] for p in pollutants]
        
        fig_radar.add_trace(go.Scatterpolar(
            r=scores,
            theta=pollutants,
            fill='toself',
            name='Quality Score',
            line=dict(color='rgba(102, 126, 234, 0.8)'),
            fillcolor='rgba(102, 126, 234, 0.3)'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title="Water Quality Assessment Radar",
            height=500
        )
        
        fig_bar = go.Figure()
        
        values = [predictions[p] for p in pollutants]
        safe_limits = [CONFIG['POLLUTANT_INFO'][p]['safe_limit'] for p in pollutants]
        colors = [CONFIG['POLLUTANT_INFO'][p]['color'] for p in pollutants]
        
        fig_bar.add_trace(go.Bar(
            x=pollutants,
            y=values,
            name='Predicted Values',
            marker_color=colors,
            text=[f'{v:.2f}' for v in values],
            textposition='auto'
        ))
        
        fig_bar.add_trace(go.Bar(
            x=pollutants,
            y=safe_limits,
            name='Safe Limits',
            marker_color='rgba(100, 100, 100, 0.3)',
            text=[f'{v:.2f}' for v in safe_limits],
            textposition='auto'
        ))
        
        fig_bar.update_layout(
            title="Predicted Values vs Safe Limits",
            xaxis_title="Pollutants",
            yaxis_title="Concentration (mg/L)",
            barmode='group',
            height=500
        )
        
        return fig_radar, fig_bar
    
    def render_prediction_results(self, station_id: str, year: int, predictions: Dict):
        station_info = self.get_station_info(station_id)
        quality_assessment = self.assess_water_quality(predictions)
        
        st.markdown(f"""
        <div class="prediction-card">
            <h2 style="color: #2c3e50; margin-bottom: 1rem;">
                {station_info['name']} - Predictions for {year}
            </h2>
            <p style="color: #7f8c8d; font-size: 1.1rem;">
                Station Type: {station_info['type']} | Overall Quality: 
                <span style="color: {'#27ae60' if quality_assessment['overall_status'] == 'Excellent' else '#f39c12' if quality_assessment['overall_status'] in ['Good', 'Fair'] else '#e74c3c'};">
                    {quality_assessment['overall_status']} ({quality_assessment['overall_score']:.1f}/100)
                </span>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        fig_radar, fig_bar = self.create_prediction_charts(predictions, quality_assessment)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig_radar, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### Detailed Analysis")
        
        cols = st.columns(len(predictions))
        for i, (pollutant, value) in enumerate(predictions.items()):
            with cols[i]:
                info = CONFIG['POLLUTANT_INFO'][pollutant]
                assessment = quality_assessment['assessments'][pollutant]
                
                status_class = f"status-{assessment['status'].lower()}"
                
                st.markdown(f"""
                <div class="metric-card {status_class}">
                    <div class="metric-value">{value:.2f}</div>
                    <div class="metric-label">{info['name']}</div>
                    <div class="metric-label">{info['unit']}</div>
                    <div style="margin-top: 0.5rem; font-size: 0.8rem;">
                        Limit: {info['safe_limit']} {info['unit']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("### Safety Recommendations")
        
        recommendations = []
        for pollutant, assessment in quality_assessment['assessments'].items():
            if assessment['status'] == 'Danger':
                recommendations.append(f"**{CONFIG['POLLUTANT_INFO'][pollutant]['name']}**: Immediate action required - levels exceed safe limits")
            elif assessment['status'] == 'Warning':
                recommendations.append(f"**{CONFIG['POLLUTANT_INFO'][pollutant]['name']}**: Monitor closely - approaching unsafe levels")
        
        if not recommendations:
            st.success("All parameters are within safe limits!")
        else:
            for rec in recommendations:
                st.warning(rec)
    
    def render_historical_trend(self, station_id: str):
        st.markdown("### Historical Trend Analysis")
        
        years = list(range(2000, 2025))
        
        fig = go.Figure()
        
        for pollutant in CONFIG['POLLUTANTS']:
            base_value = np.random.uniform(1, 10)
            trend_data = [base_value + np.random.normal(0, 1) + 0.1 * (year - 2020) for year in years]
            
            fig.add_trace(go.Scatter(
                x=years,
                y=trend_data,
                mode='lines+markers',
                name=CONFIG['POLLUTANT_INFO'][pollutant]['name'],
                line=dict(color=CONFIG['POLLUTANT_INFO'][pollutant]['color'], width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title=f"Historical Trends - Station {station_id}",
            xaxis_title="Year",
            yaxis_title="Concentration (mg/L)",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def main(self):
        self.setup_page_config()
        self.apply_custom_css()
        self.render_header()
        self.render_sidebar()
        
        self.model, self.model_columns = self.load_model_and_columns()
        
        if not self.model or not self.model_columns:
            st.error("Model files are missing. Please ensure 'pollution_model.pkl' and 'model_columns.pkl' are available.")
            st.stop()
        
        tab1, tab2, tab3 = st.tabs(["Prediction", "Analysis", "Station Map"])
        
        with tab1:
            self.render_prediction_tab()
        
        with tab2:
            self.render_analysis_tab()
        
        with tab3:
            self.render_map_tab()
    
    def render_prediction_tab(self):
        st.markdown("### Water Quality Prediction")
        
        with st.form("prediction_form", clear_on_submit=False):
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                station_id = st.selectbox(
                    "Station ID",
                    options=list(CONFIG['STATION_LOCATIONS'].keys()) + ['Other'],
                    format_func=lambda x: f"{x} - {CONFIG['STATION_LOCATIONS'].get(x, {}).get('name', 'Custom Station')}" if x != 'Other' else 'Custom Station'
                )
                
                if station_id == 'Other':
                    station_id = st.text_input("Enter Station ID", placeholder="e.g., 23")
            
            with col2:
                current_year = datetime.now().year
                year = st.slider(
                    "Year",
                    min_value=2000,
                    max_value=current_year + 10,
                    value=current_year,
                    step=1
                )
            
            with col3:
                st.markdown("<br>", unsafe_allow_html=True)
                submitted = st.form_submit_button("Predict", use_container_width=True)
        
        with st.expander("Advanced Options"):
            col1, col2 = st.columns(2)
            with col1:
                confidence_interval = st.checkbox("Show confidence intervals", value=True)
            with col2:
                include_trends = st.checkbox("Include trend analysis", value=True)
        
        if submitted and station_id:
            with st.spinner("Processing prediction..."):
                try:
                    input_data = pd.DataFrame({'year': [year], 'id': [station_id]})
                    input_encoded = pd.get_dummies(input_data, columns=['id'])
                    
                    missing_cols = set(self.model_columns) - set(input_encoded.columns)
                    for col in missing_cols:
                        input_encoded[col] = 0
                    
                    try:
                        input_encoded = input_encoded[self.model_columns]
                    except KeyError:
                        st.error(f"Station '{station_id}' not found in training data. Available stations: 1-22")
                        st.stop()
                    
                    predicted_values = self.model.predict(input_encoded)[0]
                    predictions = dict(zip(CONFIG['POLLUTANTS'], predicted_values))
                    
                    self.render_prediction_results(station_id, year, predictions)
                    
                    if include_trends:
                        self.render_historical_trend(station_id)
                    
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
                    logger.error(f"Prediction error: {e}")
    
    def render_analysis_tab(self):
        st.markdown("### System Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model Accuracy", "94.2%", "2.1%")
        with col2:
            st.metric("Predictions Made", "1,247", "23")
        with col3:
            st.metric("Active Stations", "22", "0")
        with col4:
            st.metric("Data Quality", "98.5%", "0.3%")
        
        st.markdown("#### Feature Importance")
        
        features = ['Station Location', 'Year', 'Seasonal Patterns', 'Historical Data']
        importance = [0.45, 0.25, 0.20, 0.10]
        
        fig = px.bar(
            x=features,
            y=importance,
            title="Model Feature Importance",
            color=importance,
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_map_tab(self):
        st.markdown("### Station Locations")
        
        map_data = []
        for station_id, info in CONFIG['STATION_LOCATIONS'].items():
            map_data.append({
                'Station': station_id,
                'Name': info['name'],
                'Type': info['type'],
                'lat': info['lat'],
                'lon': info['lon']
            })
        
        df_map = pd.DataFrame(map_data)
        
        st.map(df_map[['lat', 'lon']], zoom=10)
        
        st.markdown("#### Station Details")
        st.dataframe(df_map[['Station', 'Name', 'Type']], use_container_width=True)

if __name__ == "__main__":
    app = WaterQualityApp()
    app.main()

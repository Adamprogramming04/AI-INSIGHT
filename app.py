import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.utils import PlotlyJSONEncoder
import json
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB
app.secret_key = 'plasman_ai_ultimate_2025_render_deployment'

# Create upload directory
os.makedirs('uploads', exist_ok=True)

class AdvancedFinancialAI:
    def __init__(self):
        self.data = None
        self.all_sheets = {}
        self.insights = []
        self.predictions = {}
        self.risk_matrix = {}
        self.financial_health_score = 0
        self.data_quality_score = 0
        self.business_units = []
        self.kpis = {}
        
    def load_excel_data(self, file_path):
        """Load and process Excel data"""
        try:
            excel_data = pd.read_excel(
                file_path, 
                sheet_name=None, 
                engine='openpyxl'
            )
            
            self.all_sheets = excel_data
            
            # Find the best sheet
            best_sheet = self._find_best_sheet(excel_data)
            self.data = excel_data[best_sheet].copy()
            
            # Clean data
            self._clean_data()
            
            # Extract business units
            self._extract_business_units()
            
            return True
            
        except Exception as e:
            print(f"Error loading Excel: {e}")
            return False
    
    def _find_best_sheet(self, excel_data):
        """Find the most relevant sheet"""
        financial_keywords = ['cash', 'flow', 'revenue', 'profit', 'balance', 'forecast']
        best_score = 0
        best_sheet = list(excel_data.keys())[0]
        
        for name, df in excel_data.items():
            score = 0
            # Score based on name
            for keyword in financial_keywords:
                if keyword in name.lower():
                    score += 10
            
            # Score based on data
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            data_density = df.count().sum() / (df.shape[0] * df.shape[1]) if df.shape[0] * df.shape[1] > 0 else 0
            
            score += numeric_cols * 2 + data_density * 20 + df.shape[0] * 0.1
            
            if score > best_score:
                best_score = score
                best_sheet = name
        
        return best_sheet
    
    def _clean_data(self):
        """Clean and prepare data"""
        # Clean column names
        self.data.columns = [str(col).strip().replace('\n', ' ').replace('\r', ' ') 
                           for col in self.data.columns]
        
        # Remove empty rows/columns
        self.data = self.data.dropna(how='all').dropna(axis=1, how='all')
        
        # Convert numeric columns
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                # Try to convert to numeric
                numeric_series = pd.to_numeric(self.data[col], errors='coerce')
                if numeric_series.count() > len(self.data) * 0.5:
                    self.data[col] = numeric_series
        
        # Calculate data quality
        total_cells = self.data.shape[0] * self.data.shape[1]
        non_null_cells = self.data.count().sum()
        self.data_quality_score = non_null_cells / total_cells if total_cells > 0 else 0
    
    def _extract_business_units(self):
        """Extract business units from sheet names"""
        units = ['PAGE', 'PASI', 'PACA', 'PARA', 'PAST', 'PAGO']
        for unit in units:
            if any(unit in sheet_name.upper() for sheet_name in self.all_sheets.keys()):
                if unit not in self.business_units:
                    self.business_units.append(unit)
    
    def generate_insights(self):
        """Generate comprehensive insights"""
        self.insights = []
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        # Cash flow analysis
        cash_cols = [col for col in numeric_cols if 'cash' in col.lower() or 'flow' in col.lower()]
        for col in cash_cols:
            self._analyze_cash_flow(col)
        
        # Risk assessment
        for col in numeric_cols[:5]:
            self._analyze_risk(col)
        
        # Trend analysis
        for col in numeric_cols[:3]:
            self._analyze_trends(col)
        
        # Health assessment
        self._calculate_health_score()
        
        return self.insights
    
    def _analyze_cash_flow(self, column):
        """Analyze cash flow for a specific column"""
        values = self.data[column].dropna()
        if len(values) < 3:
            return
        
        trend = self._calculate_trend(values)
        volatility = values.std() / abs(values.mean()) if values.mean() != 0 else 0
        
        insight_type = 'success' if trend > 0.1 else 'warning' if trend < -0.1 else 'info'
        
        self.insights.append({
            'type': insight_type,
            'category': 'cash_flow',
            'title': f'Cash Flow Analysis: {column}',
            'message': f'Trend: {trend:.1%}, Volatility: {volatility:.1%}',
            'priority': 'high' if abs(trend) > 0.3 else 'medium',
            'data': {
                'column': column,
                'trend': trend,
                'volatility': volatility,
                'current': values.iloc[-1],
                'average': values.mean()
            }
        })
    
    def _analyze_risk(self, column):
        """Risk analysis for a column"""
        values = self.data[column].dropna()
        if len(values) < 5:
            return
        
        var_95 = np.percentile(values, 5)
        risk_score = self._calculate_risk_score(values)
        
        if risk_score > 0.7:
            self.insights.append({
                'type': 'error',
                'category': 'risk',
                'title': f'High Risk Alert: {column}',
                'message': f'Risk Score: {risk_score:.2f}, VaR(95%): {var_95:,.0f}',
                'priority': 'high',
                'data': {
                    'column': column,
                    'risk_score': risk_score,
                    'var_95': var_95
                }
            })
    
    def _analyze_trends(self, column):
        """Trend analysis"""
        values = self.data[column].dropna()
        if len(values) < 3:
            return
        
        trend = self._calculate_trend(values)
        
        if abs(trend) > 0.2:
            direction = 'positive' if trend > 0 else 'negative'
            self.insights.append({
                'type': 'success' if trend > 0 else 'warning',
                'category': 'trend',
                'title': f'Strong {direction.title()} Trend: {column}',
                'message': f'{direction.title()} trend of {abs(trend):.1%} detected',
                'priority': 'medium',
                'data': {
                    'column': column,
                    'trend': trend,
                    'direction': direction
                }
            })
    
    def _calculate_health_score(self):
        """Calculate overall financial health score"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            self.financial_health_score = 0.5
            return
        
        scores = []
        
        for col in numeric_cols:
            values = self.data[col].dropna()
            if len(values) > 1:
                trend = self._calculate_trend(values)
                stability = 1 - min(1, values.std() / abs(values.mean()) if values.mean() != 0 else 1)
                score = (trend + 1) / 2 * 0.5 + stability * 0.5
                scores.append(max(0, min(1, score)))
        
        self.financial_health_score = np.mean(scores) if scores else 0.5
        
        # Add health insight
        health_level = 'Excellent' if self.financial_health_score > 0.8 else 'Good' if self.financial_health_score > 0.6 else 'Fair' if self.financial_health_score > 0.4 else 'Poor'
        
        self.insights.append({
            'type': 'success' if self.financial_health_score > 0.6 else 'warning' if self.financial_health_score > 0.3 else 'error',
            'category': 'health',
            'title': f'Financial Health: {health_level}',
            'message': f'Overall health score: {self.financial_health_score:.1%}',
            'priority': 'high',
            'data': {
                'score': self.financial_health_score,
                'level': health_level
            }
        })
    
    def _calculate_trend(self, values):
        """Calculate trend strength"""
        if len(values) < 2:
            return 0
        
        try:
            X = np.arange(len(values)).reshape(-1, 1)
            y = values.values
            model = LinearRegression().fit(X, y)
            trend_coef = model.coef_[0]
            normalized_trend = trend_coef / (values.std() if values.std() != 0 else 1)
            return max(-1, min(1, normalized_trend))
        except:
            return 0
    
    def _calculate_risk_score(self, values):
        """Calculate composite risk score"""
        try:
            volatility = values.std() / abs(values.mean()) if values.mean() != 0 else 1
            negative_ratio = (values < 0).sum() / len(values)
            
            risk_score = min(1, volatility) * 0.6 + negative_ratio * 0.4
            return max(0, min(1, risk_score))
        except:
            return 0.5
    
    def generate_predictions(self):
        """Generate predictions for key metrics"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:3]:  # Top 3 columns
            values = self.data[col].dropna()
            if len(values) >= 5:
                try:
                    forecast = self._simple_forecast(values, periods=3)
                    if forecast is not None and len(forecast) > 0:
                        self.predictions[col] = {
                            'forecast': forecast.tolist(),
                            'current': values.iloc[-1],
                            'confidence': 0.75
                        }
                except:
                    continue
    
    def _simple_forecast(self, values, periods=3):
        """Simple forecasting using linear regression"""
        try:
            X = np.arange(len(values)).reshape(-1, 1)
            y = values.values
            
            model = LinearRegression().fit(X, y)
            
            future_X = np.arange(len(values), len(values) + periods).reshape(-1, 1)
            forecast = model.predict(future_X)
            
            return forecast
        except:
            return None
    
    def create_visualizations(self, chart_type='dashboard'):
        """Create visualizations"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return None
        
        if chart_type == 'dashboard':
            return self._create_dashboard()
        elif chart_type == 'cash_flow':
            return self._create_cash_flow_chart()
        elif chart_type == 'risk':
            return self._create_risk_chart()
        elif chart_type == 'trends':
            return self._create_trends_chart()
        elif chart_type == 'predictions':
            return self._create_predictions_chart()
        else:
            return self._create_dashboard()
    
    def _create_dashboard(self):
        """Create comprehensive dashboard"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Financial Trends', 'Performance Distribution', 'Health Gauge', 'Risk Assessment'),
            specs=[
                [{"secondary_y": True}, {"type": "box"}],
                [{"type": "indicator"}, {"type": "scatter"}]
            ]
        )
        
        # Trends
        for i, col in enumerate(numeric_cols[:3]):
            values = self.data[col].dropna()
            fig.add_trace(
                go.Scatter(y=values, name=col[:15], line=dict(width=2)),
                row=1, col=1
            )
        
        # Distribution
        for col in numeric_cols[:3]:
            fig.add_trace(
                go.Box(y=self.data[col], name=col[:15]),
                row=1, col=2
            )
        
        # Health Gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=self.financial_health_score * 100,
                title={'text': "Health Score (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ]
                }
            ),
            row=2, col=1
        )
        
        # Risk scatter
        if len(numeric_cols) >= 2:
            x_data = self.data[numeric_cols[0]].dropna()
            y_data = self.data[numeric_cols[1]].dropna()
            min_len = min(len(x_data), len(y_data))
            
            fig.add_trace(
                go.Scatter(
                    x=x_data[:min_len], 
                    y=y_data[:min_len], 
                    mode='markers',
                    name='Risk Analysis'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title='PLASMAN AB - Financial Intelligence Dashboard',
            height=800,
            template='plotly_dark',
            showlegend=True
        )
        
        return json.dumps(fig, cls=PlotlyJSONEncoder)
    
    def _create_cash_flow_chart(self):
        """Create cash flow specific chart"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        cash_cols = [col for col in numeric_cols if 'cash' in col.lower() or 'flow' in col.lower()]
        
        if not cash_cols:
            cash_cols = numeric_cols[:2]
        
        fig = go.Figure()
        
        for col in cash_cols:
            values = self.data[col].dropna()
            fig.add_trace(go.Scatter(
                y=values,
                name=col,
                line=dict(width=3),
                mode='lines+markers'
            ))
        
        fig.update_layout(
            title='Cash Flow Analysis',
            xaxis_title='Period',
            yaxis_title='Amount',
            template='plotly_dark',
            height=600
        )
        
        return json.dumps(fig, cls=PlotlyJSONEncoder)
    
    def _create_risk_chart(self):
        """Create risk assessment chart"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        risk_data = []
        for col in numeric_cols[:5]:
            values = self.data[col].dropna()
            if len(values) > 0:
                risk_score = self._calculate_risk_score(values)
                volatility = values.std() / abs(values.mean()) if values.mean() != 0 else 0
                risk_data.append({
                    'metric': col[:15],
                    'risk_score': risk_score,
                    'volatility': volatility
                })
        
        if risk_data:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[d['risk_score'] for d in risk_data],
                y=[d['volatility'] for d in risk_data],
                text=[d['metric'] for d in risk_data],
                mode='markers+text',
                marker=dict(
                    size=15,
                    color=[d['risk_score'] for d in risk_data],
                    colorscale='Reds',
                    showscale=True
                )
            ))
            
            fig.update_layout(
                title='Risk Assessment Matrix',
                xaxis_title='Risk Score',
                yaxis_title='Volatility',
                template='plotly_dark',
                height=600
            )
            
            return json.dumps(fig, cls=PlotlyJSONEncoder)
        
        return None
    
    def _create_trends_chart(self):
        """Create trends analysis chart"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        trend_data = []
        for col in numeric_cols[:6]:
            values = self.data[col].dropna()
            if len(values) > 2:
                trend = self._calculate_trend(values)
                trend_data.append({'metric': col[:15], 'trend': trend})
        
        if trend_data:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[d['metric'] for d in trend_data],
                y=[d['trend'] for d in trend_data],
                marker_color=['green' if d['trend'] > 0 else 'red' for d in trend_data]
            ))
            
            fig.update_layout(
                title='Trend Analysis',
                xaxis_title='Metrics',
                yaxis_title='Trend Strength',
                template='plotly_dark',
                height=600
            )
            
            return json.dumps(fig, cls=PlotlyJSONEncoder)
        
        return None
    
    def _create_predictions_chart(self):
        """Create predictions chart"""
        if not self.predictions:
            return None
        
        fig = go.Figure()
        
        for metric, pred_data in list(self.predictions.items())[:3]:
            if 'forecast' in pred_data:
                current_data = self.data[metric].dropna()
                forecast = pred_data['forecast']
                
                # Historical data
                fig.add_trace(go.Scatter(
                    y=current_data,
                    name=f'{metric} (Historical)',
                    line=dict(width=2)
                ))
                
                # Forecast
                future_x = list(range(len(current_data), len(current_data) + len(forecast)))
                fig.add_trace(go.Scatter(
                    x=future_x,
                    y=forecast,
                    name=f'{metric} (Forecast)',
                    line=dict(dash='dash', width=2)
                ))
        
        fig.update_layout(
            title='AI Predictions & Forecasts',
            xaxis_title='Period',
            yaxis_title='Value',
            template='plotly_dark',
            height=600
        )
        
        return json.dumps(fig, cls=PlotlyJSONEncoder)
    
    def generate_chat_response(self, message):
        """Generate AI chat responses"""
        message_lower = message.lower()
        
        if 'cash flow' in message_lower:
            return self._cash_flow_response()
        elif 'risk' in message_lower:
            return self._risk_response()
        elif 'predict' in message_lower or 'forecast' in message_lower:
            return self._prediction_response()
        elif 'health' in message_lower:
            return self._health_response()
        elif 'trend' in message_lower:
            return self._trend_response()
        else:
            return self._general_response()
    
    def _cash_flow_response(self):
        cash_insights = [i for i in self.insights if i.get('category') == 'cash_flow']
        if cash_insights:
            insight = cash_insights[0]
            data = insight.get('data', {})
            return f"Cash Flow Analysis for {data.get('column', 'N/A')}:\n- Current: {data.get('current', 0):,.0f}\n- Average: {data.get('average', 0):,.0f}\n- Trend: {data.get('trend', 0):.1%}\n- Volatility: {data.get('volatility', 0):.1%}"
        return "No cash flow data available for analysis."
    
    def _risk_response(self):
        risk_insights = [i for i in self.insights if i.get('category') == 'risk']
        if risk_insights:
            return f"Risk Assessment: {len(risk_insights)} risk factors identified. Highest risk: {risk_insights[0].get('title', 'N/A')}"
        return "Risk analysis shows manageable risk levels across all metrics."
    
    def _prediction_response(self):
        if self.predictions:
            response = "AI Predictions:\n"
            for metric, data in list(self.predictions.items())[:2]:
                forecast = data.get('forecast', [])
                if forecast:
                    response += f"- {metric}: Next value predicted at {forecast[0]:,.0f}\n"
            return response
        return "Insufficient data for reliable predictions. Need more historical data points."
    
    def _health_response(self):
        health_insight = next((i for i in self.insights if i.get('category') == 'health'), None)
        if health_insight:
            data = health_insight.get('data', {})
            return f"Financial Health Assessment:\n- Score: {data.get('score', 0):.1%}\n- Level: {data.get('level', 'Unknown')}\n- {health_insight.get('message', '')}"
        return f"Financial Health Score: {self.financial_health_score:.1%}"
    
    def _trend_response(self):
        trend_insights = [i for i in self.insights if i.get('category') == 'trend']
        if trend_insights:
            response = "Trend Analysis:\n"
            for insight in trend_insights[:3]:
                data = insight.get('data', {})
                response += f"- {data.get('column', 'N/A')}: {data.get('direction', 'stable')} trend ({data.get('trend', 0):.1%})\n"
            return response
        return "No significant trends detected in the current data."
    
    def _general_response(self):
        if not self.data:
            return "Upload financial data to begin advanced AI analysis including cash flow forecasting, risk assessment, and trend analysis."
        
        return f"Financial Intelligence Summary:\n- Records: {len(self.data):,}\n- Metrics: {len(self.data.columns)}\n- Health Score: {self.financial_health_score:.1%}\n- Insights: {len(self.insights)}\n\nAsk me about cash flow, risk, predictions, or trends for detailed analysis."

# Initialize AI system
financial_ai = AdvancedFinancialAI()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        if file and file.filename.lower().endswith(('.xlsx', '.xls', '.xlsm', '.xlsb')):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process with AI
            if financial_ai.load_excel_data(filepath):
                insights = financial_ai.generate_insights()
                financial_ai.generate_predictions()
                dashboard_chart = financial_ai.create_visualizations('dashboard')
                
                summary_stats = {
                    'total_rows': len(financial_ai.data),
                    'total_columns': len(financial_ai.data.columns),
                    'insights_count': len(insights),
                    'health_score': financial_ai.financial_health_score,
                    'data_quality': financial_ai.data_quality_score,
                    'business_units': financial_ai.business_units
                }
                
                return jsonify({
                    'success': True,
                    'insights': insights,
                    'chart': dashboard_chart,
                    'filename': filename,
                    'summary': summary_stats,
                    'predictions': financial_ai.predictions
                })
            else:
                return jsonify({'error': 'Failed to analyze file'})
        
        return jsonify({'error': 'Invalid file format. Please upload Excel files only.'})
    
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message', '')
        response = financial_ai.generate_chat_response(user_message)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'response': f'Error: {str(e)}'})

@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        chart_type = request.json.get('type', 'dashboard')
        chart = financial_ai.create_visualizations(chart_type)
        
        if chart:
            return jsonify({'chart': chart})
        else:
            return jsonify({'error': f'Unable to generate {chart_type} chart'})
    except Exception as e:
        return jsonify({'error': f'Chart generation failed: {str(e)}'})

@app.route('/data_view', methods=['POST'])
def data_view():
    try:
        view_type = request.json.get('type', 'summary')
        
        if financial_ai.data is None:
            return jsonify({'error': 'No data loaded'})
        
        if view_type == 'summary':
            numeric_cols = financial_ai.data.select_dtypes(include=[np.number]).columns
            summary = {
                'total_records': len(financial_ai.data),
                'total_columns': len(financial_ai.data.columns),
                'numeric_columns': len(numeric_cols),
                'data_quality': financial_ai.data_quality_score,
                'column_names': list(financial_ai.data.columns)[:10],  # First 10 columns
                'sample_data': financial_ai.data.head(5).to_dict('records')
            }
            return jsonify({'data': summary})
        
        elif view_type == 'columns':
            columns_info = []
            for col in financial_ai.data.columns:
                col_info = {
                    'name': col,
                    'type': str(financial_ai.data[col].dtype),
                    'non_null': financial_ai.data[col].count(),
                    'total': len(financial_ai.data[col])
                }
                columns_info.append(col_info)
            return jsonify({'data': columns_info})
        
        elif view_type == 'statistics':
            numeric_data = financial_ai.data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) > 0:
                stats_data = numeric_data.describe().to_dict()
                return jsonify({'data': stats_data})
            else:
                return jsonify({'error': 'No numeric data available'})
        
        else:
            return jsonify({'error': 'Invalid view type'})
    
    except Exception as e:
        return jsonify({'error': f'Data view failed: {str(e)}'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

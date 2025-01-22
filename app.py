import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import plotly.graph_objects as go
from typing import List, Tuple, Dict

class CryptoAnalyzer:
    def __init__(self):
        # Established cryptocurrencies to use as reference
        self.established_cryptos = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']
        self.success_metrics = ['volatility', 'volume_trend', 'price_momentum', 'market_correlation']
        
    def fetch_historical_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Fetch historical data for a cryptocurrency"""
        try:
            crypto = yf.Ticker(symbol)
            df = crypto.history(period=period)
            return df
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def calculate_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate success metrics for a cryptocurrency"""
        metrics = {}
        
        if len(df) < 30:  # Ensure enough data points
            return None
            
        # Volatility (30-day rolling standard deviation of returns)
        daily_returns = df['Close'].pct_change()
        metrics['volatility'] = daily_returns.std() * np.sqrt(252)  # Annualized
        
        # Volume trend (linear regression slope of volume)
        volume_trend = np.polyfit(range(len(df)), df['Volume'], 1)[0]
        metrics['volume_trend'] = volume_trend
        
        # Price momentum (ratio of current price to 30-day moving average)
        ma30 = df['Close'].rolling(window=30).mean()
        metrics['price_momentum'] = df['Close'].iloc[-1] / ma30.iloc[-1]
        
        # Market correlation (correlation with BTC)
        btc_data = self.fetch_historical_data('BTC-USD', period="1y")
        if not btc_data.empty and len(btc_data) == len(df):
            metrics['market_correlation'] = df['Close'].corr(btc_data['Close'])
        else:
            metrics['market_correlation'] = 0
            
        return metrics

    def analyze_new_crypto(self, symbol: str) -> Dict[str, float]:
        """Analyze a new cryptocurrency and compare it to established patterns"""
        # Fetch data for new crypto
        new_crypto_data = self.fetch_historical_data(symbol)
        if new_crypto_data.empty:
            return None
            
        # Calculate metrics for new crypto
        new_metrics = self.calculate_metrics(new_crypto_data)
        if not new_metrics:
            return None
            
        # Get historical patterns from established cryptos
        established_patterns = self.get_established_patterns()
        
        # Compare metrics and calculate success probability
        success_probability = self.calculate_success_probability(new_metrics, established_patterns)
        
        return {
            'metrics': new_metrics,
            'success_probability': success_probability
        }
        
    def get_established_patterns(self) -> List[Dict[str, float]]:
        """Get patterns from established cryptocurrencies"""
        patterns = []
        for crypto in self.established_cryptos:
            data = self.fetch_historical_data(crypto)
            if not data.empty:
                metrics = self.calculate_metrics(data)
                if metrics:
                    patterns.append(metrics)
        return patterns
        
    def calculate_success_probability(self, 
                                   new_metrics: Dict[str, float], 
                                   established_patterns: List[Dict[str, float]]) -> float:
        """Calculate success probability based on similarity to established patterns"""
        if not established_patterns:
            return 0.0
            
        # Normalize metrics
        scaler = StandardScaler()
        established_metrics_array = []
        for pattern in established_patterns:
            established_metrics_array.append([pattern[metric] for metric in self.success_metrics])
        
        established_metrics_normalized = scaler.fit_transform(established_metrics_array)
        new_metrics_normalized = scaler.transform([[new_metrics[metric] for metric in self.success_metrics]])
        
        # Calculate similarity scores
        similarities = []
        for established_metric in established_metrics_normalized:
            similarity = 1 / (1 + np.linalg.norm(new_metrics_normalized[0] - established_metric))
            similarities.append(similarity)
            
        # Return average similarity as success probability
        return np.mean(similarities)

def main():
    st.title("Cryptocurrency Success Analyzer")
    st.write("""
    This app analyzes new cryptocurrencies and predicts their potential success based on patterns
    observed in established cryptocurrencies.
    """)
    
    # Initialize analyzer
    analyzer = CryptoAnalyzer()
    
    # Input for new crypto symbol
    crypto_symbol = st.text_input("Enter cryptocurrency symbol (e.g., DOGE-USD):")
    
    if st.button("Analyze"):
        if crypto_symbol:
            with st.spinner("Analyzing cryptocurrency..."):
                analysis_result = analyzer.analyze_new_crypto(crypto_symbol)
                
                if analysis_result:
                    st.subheader("Analysis Results")
                    
                    # Display metrics
                    metrics = analysis_result['metrics']
                    st.write("### Key Metrics")
                    for metric, value in metrics.items():
                        st.write(f"{metric.replace('_', ' ').title()}: {value:.4f}")
                    
                    # Display success probability
                    success_prob = analysis_result['success_probability']
                    st.write("### Success Probability")
                    st.write(f"Based on historical patterns: {success_prob:.2%}")
                    
                    # Create gauge chart for success probability
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = success_prob * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Success Probability"},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 33], 'color': "lightgray"},
                                {'range': [33, 66], 'color': "gray"},
                                {'range': [66, 100], 'color': "darkgray"}
                            ],
                        }
                    ))
                    st.plotly_chart(fig)
                    
                    # Disclaimer
                    st.warning("""
                    Please note: This analysis is based on historical patterns and should not be 
                    considered as financial advice. Cryptocurrency investments carry high risk, 
                    and past performance does not guarantee future results.
                    """)
                else:
                    st.error("Unable to analyze the cryptocurrency. Please check the symbol and try again.")
        else:
            st.warning("Please enter a cryptocurrency symbol.")

if __name__ == "__main__":
    main()
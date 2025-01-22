import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import requests
from textblob import TextBlob
import ccxt
from scipy import stats
import ta

class AdvancedCryptoAnalyzer:
    def __init__(self):
        self.established_cryptos = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD']
        self.success_metrics = [
            'volatility', 'volume_trend', 'price_momentum', 'market_correlation',
            'liquidity_depth', 'development_activity', 'social_sentiment',
            'whale_activity', 'network_growth', 'token_distribution'
        ]
        self.exchange = ccxt.binance()
        
    def fetch_order_book_data(self, symbol: str) -> Dict:
        """Fetch order book data to analyze market depth"""
        try:
            order_book = self.exchange.fetch_order_book(symbol)
            return order_book
        except:
            return None

    def calculate_liquidity_depth(self, order_book: Dict) -> float:
        """Calculate liquidity depth based on order book"""
        if not order_book:
            return 0
            
        bids = np.array(order_book['bids'])
        asks = np.array(order_book['asks'])
        
        # Calculate bid-ask spread and depth
        spread = (asks[0][0] - bids[0][0]) / bids[0][0]
        depth = np.sum(bids[:10, 1]) + np.sum(asks[:10, 1])
        
        return (1 / spread) * depth

    def fetch_github_activity(self, repo_url: str) -> float:
        """Fetch GitHub development activity metrics"""
        try:
            # You'll need to add GitHub API token for production use
            headers = {'Authorization': 'token YOUR_GITHUB_TOKEN'}
            api_url = repo_url.replace('github.com', 'api.github.com/repos')
            
            response = requests.get(f"{api_url}/stats/commit_activity", headers=headers)
            commit_data = response.json()
            
            # Calculate weekly average commits
            weekly_commits = [week['total'] for week in commit_data]
            return np.mean(weekly_commits)
        except:
            return 0

    def analyze_social_sentiment(self, symbol: str) -> float:
        """Analyze social media sentiment"""
        try:
            # Replace with your preferred social media API
            tweets = self.fetch_crypto_tweets(symbol)
            reddit_posts = self.fetch_reddit_posts(symbol)
            
            # Analyze sentiment
            sentiments = []
            for text in tweets + reddit_posts:
                blob = TextBlob(text)
                sentiments.append(blob.sentiment.polarity)
                
            return np.mean(sentiments)
        except:
            return 0

    def analyze_whale_activity(self, df: pd.DataFrame) -> float:
        """Analyze large transaction patterns"""
        large_transactions = df[df['Volume'] > df['Volume'].quantile(0.95)]
        return len(large_transactions) / len(df)

    def calculate_network_metrics(self, symbol: str) -> Dict[str, float]:
        """Calculate on-chain metrics"""
        try:
            # Integrate with blockchain explorer APIs for production
            metrics = {
                'active_addresses': 0,
                'transaction_count': 0,
                'average_transaction_value': 0
            }
            return metrics
        except:
            return None

    def analyze_token_distribution(self, symbol: str) -> float:
        """Analyze token distribution and concentration"""
        try:
            # Integrate with blockchain explorer APIs for production
            holder_data = self.fetch_token_holders(symbol)
            gini_coefficient = self.calculate_gini_coefficient(holder_data)
            return gini_coefficient
        except:
            return 0

    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate advanced technical indicators"""
        indicators = {}
        
        # Trend indicators
        indicators['macd'] = ta.trend.macd_diff(df['Close'])
        indicators['rsi'] = ta.momentum.rsi(df['Close'])
        indicators['bollinger_hband'] = ta.volatility.bollinger_hband(df['Close'])
        indicators['bollinger_lband'] = ta.volatility.bollinger_lband(df['Close'])
        
        # Volume indicators
        indicators['volume_adi'] = ta.volume.acc_dist_index(df['High'], df['Low'], df['Close'], df['Volume'])
        indicators['volume_obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        
        # Momentum indicators
        indicators['awesome_oscillator'] = ta.momentum.awesome_oscillator(df['High'], df['Low'])
        indicators['kama'] = ta.momentum.kama(df['Close'])
        
        return indicators

    def calculate_risk_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics"""
        daily_returns = df['Close'].pct_change()
        
        risk_metrics = {
            'sharpe_ratio': self.calculate_sharpe_ratio(daily_returns),
            'sortino_ratio': self.calculate_sortino_ratio(daily_returns),
            'max_drawdown': self.calculate_max_drawdown(df['Close']),
            'var_95': self.calculate_value_at_risk(daily_returns, 0.95)
        }
        
        return risk_metrics

    def calculate_market_impact(self, df: pd.DataFrame, order_book: Dict) -> float:
        """Calculate potential market impact of trades"""
        if not order_book:
            return 0
            
        avg_daily_volume = df['Volume'].mean()
        liquidity_depth = self.calculate_liquidity_depth(order_book)
        
        return avg_daily_volume / liquidity_depth

    def predict_trend(self, df: pd.DataFrame) -> Dict[str, float]:
        """Predict short-term price trend"""
        # Implement your preferred prediction model here
        # This is a simplified example using linear regression
        X = np.arange(len(df)).reshape(-1, 1)
        y = df['Close'].values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(X.flatten(), y)
        
        prediction = {
            'slope': slope,
            'r_squared': r_value ** 2,
            'confidence': 1 - p_value
        }
        
        return prediction

    def generate_trading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on multiple indicators"""
        signals = pd.DataFrame(index=df.index)
        
        # Calculate indicators
        signals['sma_20'] = df['Close'].rolling(window=20).mean()
        signals['sma_50'] = df['Close'].rolling(window=50).mean()
        signals['rsi'] = ta.momentum.rsi(df['Close'])
        
        # Generate signals
        signals['signal'] = 0
        signals.loc[
            (signals['sma_20'] > signals['sma_50']) & 
            (signals['rsi'] < 70),
            'signal'
        ] = 1
        signals.loc[
            (signals['sma_20'] < signals['sma_50']) & 
            (signals['rsi'] > 30),
            'signal'
        ] = -1
        
        return signals

def main():
    st.title("Advanced Cryptocurrency Analysis Platform")
    
    # Sidebar for analysis options
    st.sidebar.header("Analysis Options")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Technical Analysis", "On-Chain Analysis", "Market Structure", "Risk Analysis"]
    )
    
    # Initialize analyzer
    analyzer = AdvancedCryptoAnalyzer()
    
    # Input for cryptocurrency
    crypto_symbol = st.text_input("Enter cryptocurrency symbol (e.g., BTC/USDT):")
    
    if st.button("Analyze"):
        if crypto_symbol:
            with st.spinner("Performing comprehensive analysis..."):
                # Fetch data
                df = analyzer.fetch_historical_data(crypto_symbol)
                order_book = analyzer.fetch_order_book_data(crypto_symbol)
                
                if not df.empty:
                    # Display analysis based on selected type
                    if analysis_type == "Technical Analysis":
                        indicators = analyzer.calculate_technical_indicators(df)
                        signals = analyzer.generate_trading_signals(df)
                        trend_prediction = analyzer.predict_trend(df)
                        
                        # Create technical analysis dashboard
                        st.subheader("Technical Analysis Dashboard")
                        
                        # Plot price and indicators
                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(
                            x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            name="Price"
                        ))
                        
                        # Add indicators
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=indicators['bollinger_hband'],
                            name="Upper BB",
                            line=dict(color='gray', dash='dash')
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=indicators['bollinger_lband'],
                            name="Lower BB",
                            line=dict(color='gray', dash='dash'),
                            fill='tonexty'
                        ))
                        
                        st.plotly_chart(fig)
                        
                        # Display trading signals
                        st.write("### Trading Signals")
                        signal_fig = px.scatter(
                            signals,
                            x=signals.index,
                            y=df['Close'],
                            color='signal',
                            title="Trading Signals"
                        )
                        st.plotly_chart(signal_fig)
                        
                    elif analysis_type == "On-Chain Analysis":
                        network_metrics = analyzer.calculate_network_metrics(crypto_symbol)
                        whale_activity = analyzer.analyze_whale_activity(df)
                        token_distribution = analyzer.analyze_token_distribution(crypto_symbol)
                        
                        st.subheader("On-Chain Analysis")
                        if network_metrics:
                            st.write("### Network Activity")
                            for metric, value in network_metrics.items():
                                st.metric(
                                    label=metric.replace('_', ' ').title(),
                                    value=f"{value:,.2f}"
                                )
                        
                        st.write("### Whale Activity")
                        st.metric(
                            label="Large Transaction Ratio",
                            value=f"{whale_activity:.2%}"
                        )
                        
                        st.write("### Token Distribution")
                        st.metric(
                            label="Gini Coefficient",
                            value=f"{token_distribution:.4f}"
                        )
                        
                    elif analysis_type == "Market Structure":
                        market_impact = analyzer.calculate_market_impact(df, order_book)
                        liquidity_depth = analyzer.calculate_liquidity_depth(order_book)
                        
                        st.subheader("Market Structure Analysis")
                        
                        # Plot order book
                        if order_book:
                            bids = np.array(order_book['bids'])
                            asks = np.array(order_book['asks'])
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=bids[:, 0],
                                y=np.cumsum(bids[:, 1]),
                                name="Bids",
                                fill='tozeroy'
                            ))
                            fig.add_trace(go.Scatter(
                                x=asks[:, 0],
                                y=np.cumsum(asks[:, 1]),
                                name="Asks",
                                fill='tozeroy'
                            ))
                            
                            fig.update_layout(title="Order Book Depth")
                            st.plotly_chart(fig)
                            
                        st.metric(
                            label="Market Impact Score",
                            value=f"{market_impact:.4f}"
                        )
                        
                        st.metric(
                            label="Liquidity Depth",
                            value=f"{liquidity_depth:,.2f}"
                        )
                        
                    else:  # Risk Analysis
                        risk_metrics = analyzer.calculate_risk_metrics(df)
                        
                        st.subheader("Risk Analysis")
                        for metric, value in risk_metrics.items():
                            st.metric(
                                label=metric.replace('_', ' ').title(),
                                value=f"{value:.4f}"
                            )
                            
                        # Plot returns distribution
                        returns = df['Close'].pct_change().dropna()
                        fig = px.histogram(
                            returns,
                            title="Returns Distribution",
                            nbins=50
                        )
                        st.plotly_chart(fig)
                        
                    # Common metrics for all analysis types
                    st.subheader("Additional Insights")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="30-Day Return",
                            value=f"{((df['Close'].iloc[-1] / df['Close'].iloc[-30]) - 1):.2%}"
                        )
                        
                    with col2:
                        st.metric(
                            label="Volatility",
                            value=f"{df['Close'].pct_change().std() * np.sqrt(252):.2%}"
                        )
                        
                    with col3:
                        st.metric(
                            label="Volume Trend",
                            value=f"{np.polyfit(range(len(df)), df['Volume'], 1)[0]:,.0f}"
                        )
                        
                    # Disclaimer
                    st.warning("""
                    This analysis is for informational purposes only and should not be 
                    considered as financial advice. Cryptocurrency investments carry 
                    significant risks. Always conduct your own research and consider 
                    consulting with a financial advisor.
                    """)
                else:
                    st.error("Unable to fetch data. Please check the symbol and try again.")
        else:
            st.warning("Please enter a cryptocurrency symbol.")

if __name__ == "__main__":
    main()
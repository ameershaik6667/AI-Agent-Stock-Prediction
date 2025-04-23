import streamlit as st
import requests
import pandas as pd
import yfinance as yf
import json
import torch
import torch.nn as nn
import torch.optim as optim
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
import backtrader as bt
import os
import pickle
from datetime import datetime

st.title("Prototype Trading System")

symbol = st.text_input("Enter Stock Symbol:", value="AAPL")

if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None
if 'sentiment_data' not in st.session_state:
    st.session_state.sentiment_data = None

class DataCollectionAgent:
    def fetch_news_articles(self, symbol):
        api_key = "cuik1g1r01qtqfmisj9gcuik1g1r01qtqfmisja0"
        url = f"https://finnhub.io/api/v1/news?category=general&token={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            articles = response.json()
            return articles[:5]
        return []

    def collect_social_media_data(self, symbol):
        return [
            f"{symbol} is on the rise! 🚀",
            f"Worried about {symbol}'s volatility today... 😬",
            f"Market analysts predict a strong quarter for {symbol}. 📈"
        ]

    def get_financial_reports(self, symbol):
        stock = yf.Ticker(symbol)
        try:
            reports = {
                "Income Statement": stock.financials,
                "Balance Sheet": stock.balance_sheet,
                "Cash Flow": stock.cashflow
            }
            return reports
        except Exception as e:
            return f"Error fetching financial data: {str(e)}"

data_agent = DataCollectionAgent()

if st.button("Fetch News Articles"):
    articles = data_agent.fetch_news_articles(symbol)
    if articles:
        st.subheader(f"Latest News for {symbol}")
        for article in articles:
            st.write(f"**{article['headline']}** - {article['source']}")
            st.write(article['summary'])
            st.write(f"[Read more]({article['url']})")
            st.write("---")
    else:
        st.warning("No news articles found or API issue.")

if st.button("Fetch Social Media Data"):
    social_posts = data_agent.collect_social_media_data(symbol)
    if social_posts:
        st.subheader(f"Social Media Mentions for {symbol}")
        for post in social_posts:
            st.write(f"- {post}")
    else:
        st.warning("No social media data available.")

if st.button("Fetch Financial Reports"):
    reports = data_agent.get_financial_reports(symbol)
    if isinstance(reports, dict):
        st.subheader(f"Financial Reports for {symbol}")
        for key, df in reports.items():
            st.write(f"**{key}**")
            st.write(df)
            st.write("---")
    else:
        st.warning("No financial reports found or an error occurred.")

class SentimentAnalysisAgent:
    def analyze_text_sentiment(self, text):
        score = TextBlob(text).sentiment.polarity
        sentiment = "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"
        return score, sentiment

    def train_ml_sentiment_model(self, texts, labels):
        vectorizer = CountVectorizer()
        tfidf_transformer = TfidfTransformer()
        X_counts = vectorizer.fit_transform(texts)
        X_tfidf = tfidf_transformer.fit_transform(X_counts)
        X_train, X_test, y_train, y_test = train_test_split(X_tfidf, labels, test_size=0.2)
        model = MultinomialNB()
        model.fit(X_train, y_train)
        return model

sentiment_agent = SentimentAnalysisAgent()

class FeatureEngineeringAgent:
    def extract_sentiment_features(self, texts):
        return [TextBlob(text).sentiment.polarity for text in texts]

    def integrate_features_into_model(self, model, features):
        return model.predict(features)

    def implement_model_stacking(self):
        base_models = [
            ("svm", SVC(probability=True)),
            ("naive_bayes", MultinomialNB())
        ]
        stacked_model = StackingClassifier(
            estimators=base_models, final_estimator=LogisticRegression()
        )
        return stacked_model

feature_engineering_agent = FeatureEngineeringAgent()

class TradingStrategyAgent:
    def __init__(self, initial_capital=10000, stop_loss=0.95, take_profit=1.10):
        self.initial_capital = initial_capital
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def generate_trading_signal(self, sentiment_score):
        if sentiment_score > 0.2:
            return "BUY"
        elif sentiment_score < -0.2:
            return "SELL"
        return "HOLD"

    def backtest_strategy(self, data):
        capital = self.initial_capital
        position = 0
        trade_log = []
        
        for i in range(1, len(data)):
            price = data["Close"].iloc[i]
            prev_price = data["Close"].iloc[i-1]
            signal = data["Signal"].iloc[i-1]
            
            if signal == "BUY":
                position = capital / price
                capital = 0
                trade_log.append(f"BUY at {price}")
            elif signal == "SELL" and position > 0:
                capital = position * price
                position = 0
                trade_log.append(f"SELL at {price}")
            
            if position > 0 and (price < prev_price * self.stop_loss or price > prev_price * self.take_profit):
                capital = position * price
                position = 0
                trade_log.append(f"Exit due to stop-loss/take-profit at {price}")

        final_value = capital + (position * data["Close"].iloc[-1] if position > 0 else 0)
        return final_value, trade_log

    def calculate_performance_metrics(self, initial_capital, final_capital, returns):
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6)
        max_drawdown = np.min(returns)
        return {"Final Capital": final_capital, "Sharpe Ratio": sharpe_ratio, "Max Drawdown": max_drawdown}

strategy_agent = TradingStrategyAgent()

class SentimentPandasData(bt.feeds.PandasData):
    lines = ('sentiment',)
    params = (('sentiment', -1),)

class SentimentStrategy(bt.Strategy):
    params = (('stop_loss', 0.95), ('take_profit', 1.10))
    
    def __init__(self):
        self.sentiment = self.datas[0].sentiment
        self.order = None
        self.position_size = 0

    def next(self):
        if self.order:
            return
        sentiment_score = self.sentiment[0]
        price = self.datas[0].close[0]
        prev_price = self.datas[0].close[-1] if len(self.datas[0].close) > 1 else price
        
        if sentiment_score > 0.2 and not self.position:
            self.order = self.buy(size=100)
            self.position_size = 100
        elif sentiment_score < -0.2 and self.position:
            self.order = self.sell(size=self.position_size)
            self.position_size = 0
        elif self.position and (price < prev_price * self.params.stop_loss or price > prev_price * self.params.take_profit):
            self.order = self.sell(size=self.position_size)
            self.position_size = 0

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None

class EvaluationAgent:
    def __init__(self):
        self.backtest_results = None
        self.cache_dir = "yf_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def fetch_historical_data(self, symbol):
        cache_file = os.path.join(self.cache_dir, f"{symbol}_1y.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                if (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))).days < 1:
                    return data
        stock = yf.Ticker(symbol)
        try:
            data = stock.history(period="1y")
            if not data.empty:
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
                return data
            else:
                st.error(f"No data returned for {symbol}. Please check the symbol or try again later.")
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Failed to fetch data for {symbol}: {str(e)}. Please try again later or check your internet connection.")
            return pd.DataFrame()

    def fetch_sentiment_data(self, symbol, dates):
        articles = data_agent.fetch_news_articles(symbol)
        if not articles:
            st.warning("No news articles available. Using random sentiment for testing.")
            return pd.Series(np.random.uniform(-0.5, 0.5, len(dates)), index=dates)
        sentiment_scores = [sentiment_agent.analyze_text_sentiment(article['summary'])[0] for article in articles]
        avg_sentiment = np.mean(sentiment_scores)
        # Interpolate sentiment across all dates (simplified: repeat average)
        return pd.Series([avg_sentiment] * len(dates), index=dates)

    def conduct_out_of_sample_testing(self, data, train_ratio=0.8):
        train_size = int(len(data) * train_ratio)
        train_data = data[:train_size]
        test_data = data[train_size:]
        return train_data, test_data

    def analyze_performance_metrics(self, cerebro):
        returns = cerebro.runstrats[0][0].analyzers.returns.get_analysis()
        sharpe = cerebro.runstrats[0][0].analyzers.sharpe.get_analysis()
        drawdown = cerebro.runstrats[0][0].analyzers.drawdown.get_analysis()
        return {
            "Total Return": returns['rtot'],
            "Sharpe Ratio": sharpe['sharperatio'] if sharpe['sharperatio'] is not None else 0,
            "Max Drawdown": drawdown['max']['drawdown']
        }

    def forward_test(self, symbol, cash=10000):
        cerebro = bt.Cerebro()
        if st.session_state.historical_data is None:
            st.session_state.historical_data = self.fetch_historical_data(symbol)
        data = st.session_state.historical_data.copy()
        if data.empty:
            st.error("No historical data available for forward test.")
            return cerebro
        if st.session_state.sentiment_data is None:
            st.session_state.sentiment_data = self.fetch_sentiment_data(symbol, data.index)
        data["sentiment"] = st.session_state.sentiment_data
        bt_data = SentimentPandasData(
            dataname=data,
            open='Open',
            high='High',
            low='Low',
            close='Close',
            volume='Volume',
            sentiment='sentiment'
        )
        cerebro.adddata(bt_data)
        cerebro.addstrategy(SentimentStrategy)
        cerebro.broker.setcash(cash)
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.run()
        return cerebro

evaluation_agent = EvaluationAgent()

if st.button("Backtest Strategy"):
    data = pd.DataFrame({
        "Close": [100, 102, 105, 103, 107, 110],
        "Signal": ["HOLD", "BUY", "HOLD", "SELL", "BUY", "HOLD"]
    })
    final_capital, trade_log = strategy_agent.backtest_strategy(data)
    returns = np.diff(data["Close"]) / data["Close"][:-1]
    metrics = strategy_agent.calculate_performance_metrics(strategy_agent.initial_capital, final_capital, returns)

    st.write(f"Final capital after backtesting: ${final_capital:.2f}")
    st.write(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
    st.write(f"Max Drawdown: {metrics['Max Drawdown']:.2f}")
    st.subheader("Trade Log")
    for trade in trade_log:
        st.write(trade)

if st.button("Analyze News Sentiment"):
    articles = data_agent.fetch_news_articles(symbol)
    sentiment_results = []
    for article in articles:
        score, sentiment = sentiment_agent.analyze_text_sentiment(article["summary"])
        signal = strategy_agent.generate_trading_signal(score)
        sentiment_results.append((article["headline"], sentiment, signal))
        st.write(f"**{article['headline']}** - Sentiment: {sentiment} - Signal: {signal}")
        st.write("---")

if st.button("Run Backtrader Backtest"):
    cerebro = bt.Cerebro()
    if st.session_state.historical_data is None:
        st.session_state.historical_data = evaluation_agent.fetch_historical_data(symbol)
    data = st.session_state.historical_data.copy()
    if data.empty:
        st.error("No historical data available for backtest.")
    else:
        if st.session_state.sentiment_data is None:
            st.session_state.sentiment_data = evaluation_agent.fetch_sentiment_data(symbol, data.index)
        data["sentiment"] = st.session_state.sentiment_data
        bt_data = SentimentPandasData(
            dataname=data,
            open='Open',
            high='High',
            low='Low',
            close='Close',
            volume='Volume',
            sentiment='sentiment'
        )
        cerebro.adddata(bt_data)
        cerebro.addstrategy(SentimentStrategy)
        cerebro.broker.setcash(10000)
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.run()
        metrics = evaluation_agent.analyze_performance_metrics(cerebro)
        st.write(f"Backtrader Results - Total Return: {metrics['Total Return']:.2f}")
        st.write(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
        st.write(f"Max Drawdown: {metrics['Max Drawdown']:.2f}")

if st.button("Out-of-Sample Test"):
    if st.session_state.historical_data is None:
        st.session_state.historical_data = evaluation_agent.fetch_historical_data(symbol)
    data = st.session_state.historical_data.copy()
    if data.empty:
        st.error("No historical data available for out-of-sample test.")
    else:
        train_data, test_data = evaluation_agent.conduct_out_of_sample_testing(data)
        st.write(f"Training data size: {len(train_data)}")
        st.write(f"Out-of-sample test data size: {len(test_data)}")
        if st.session_state.sentiment_data is None:
            st.session_state.sentiment_data = evaluation_agent.fetch_sentiment_data(symbol, data.index)
        test_data["sentiment"] = st.session_state.sentiment_data.loc[test_data.index]
        cerebro = bt.Cerebro()
        bt_data = SentimentPandasData(
            dataname=test_data,
            open='Open',
            high='High',
            low='Low',
            close='Close',
            volume='Volume',
            sentiment='sentiment'
        )
        cerebro.adddata(bt_data)
        cerebro.addstrategy(SentimentStrategy)
        cerebro.broker.setcash(10000)
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.run()
        metrics = evaluation_agent.analyze_performance_metrics(cerebro)
        st.write(f"Out-of-Sample Results - Total Return: {metrics['Total Return']:.2f}")
        st.write(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
        st.write(f"Max Drawdown: {metrics['Max Drawdown']:.2f}")

if st.button("Forward Test"):
    cerebro = evaluation_agent.forward_test(symbol)
    if not st.session_state.historical_data.empty:
        metrics = evaluation_agent.analyze_performance_metrics(cerebro)
        st.write(f"Forward Test Results - Total Return: {metrics['Total Return']:.2f}")
        st.write(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
        st.write(f"Max Drawdown: {metrics['Max Drawdown']:.2f}")
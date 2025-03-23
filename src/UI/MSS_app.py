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

st.title("Prototype Trading System")

symbol = st.text_input("Enter Stock Symbol:", value="AAPL")

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
            
            # Stop-loss & Take-Profit logic
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



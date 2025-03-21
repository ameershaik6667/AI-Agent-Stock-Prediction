import os
import sys
import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_ta as ta
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.Data_Retrieval.data_fetcher import DataFetcher
from src.Indicators.sma import SMAIndicator

st.title("Prototype Trading System")

symbol = st.text_input("Enter Stock Symbol:", value="AAPL")

data_fetcher = DataFetcher()
data = data_fetcher.get_stock_data(symbol)

st.write(f"Original Stock Data for {symbol}:")
st.dataframe(data.tail())

if st.button("Calculate SMA"):
    period = st.number_input("Enter SMA period:", min_value=1, max_value=100, value=14)
    sma_indicator = SMAIndicator(period=period)
    data_with_sma = sma_indicator.calculate(data)
    st.write(f"Stock Data with SMA{period} for {symbol}:")
    st.dataframe(data_with_sma.tail())

if st.button("Calculate RSI"):
    period = st.number_input("Enter RSI period:", min_value=1, max_value=100, value=14)
    data[f"RSI{period}"] = ta.rsi(data['Close'], length=period)
    st.write(f"Stock Data with RSI{period} for {symbol}:")
    st.dataframe(data.tail())

class TradingStrategyAgent:
    def generate_trading_signal(self, sentiment_score):
        if sentiment_score > 0.2:
            return "BUY"
        elif sentiment_score < -0.2:
            return "SELL"
        return "HOLD"

    def backtest_strategy(self, data):
        capital = 10000
        position = 0
        for i in range(1, len(data)):
            if data["Signal"].iloc[i-1] == "BUY":
                position = capital / data["Close"].iloc[i]
                capital = 0
            elif data["Signal"].iloc[i-1] == "SELL" and position > 0:
                capital = position * data["Close"].iloc[i]
                position = 0
        return capital + (position * data["Close"].iloc[-1] if position > 0 else 0)

class DataCollectionAgent:
    def fetch_news_articles(self, symbol):
        url = f"https://finnhub.io/api/v1/news?category=general&token=cuik1g1r01qtqfmisj9gcuik1g1r01qtqfmisja0"
        response = requests.get(url)
        if response.status_code == 200:
            articles = response.json()
            return articles[:5]
        return []

    def collect_social_media_data(self, symbol):
        sample_posts = [
            f"{symbol} is on the rise!",
            f"Worried about {symbol}'s volatility today...",
            f"Market analysts predict a strong quarter for {symbol}.",
        ]
        return sample_posts

    def get_financial_reports(self, symbol):
        stock = yf.Ticker(symbol)
        return stock.financials

class SentimentAnalysisAgent:
    def analyze_text_sentiment(self, text):
        sentiment_score = TextBlob(text).sentiment.polarity
        sentiment = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
        return sentiment_score, sentiment

    def analyze_news_sentiment(self, articles):
        results = []
        for article in articles:
            score, sentiment = self.analyze_text_sentiment(article["summary"])
            results.append({"headline": article["headline"], "sentiment": sentiment, "score": score})
        return results

    def analyze_social_media_sentiment(self, posts):
        results = []
        for post in posts:
            score, sentiment = self.analyze_text_sentiment(post)
            results.append({"post": post, "sentiment": sentiment, "score": score})
        return results

data_agent = DataCollectionAgent()
sentiment_agent = SentimentAnalysisAgent()
strategy_agent = TradingStrategyAgent()

if st.button("Fetch News Articles"):
    articles = data_agent.fetch_news_articles(symbol)
    for article in articles:
        st.write(f"**{article['headline']}** - {article['source']}")
        st.write(article['summary'])
        st.write("---")

if st.button("Analyze News Sentiment"):
    articles = data_agent.fetch_news_articles(symbol)
    sentiment_results = sentiment_agent.analyze_news_sentiment(articles)
    signals = []
    for result in sentiment_results:
        signal = strategy_agent.generate_trading_signal(result["score"])
        signals.append(signal)
        st.write(f"**{result['headline']}** - Sentiment: {result['sentiment']} (Score: {result['score']:.2f}) - Signal: {signal}")
        st.write("---")

if st.button("Analyze Social Media Sentiment"):
    posts = data_agent.collect_social_media_data(symbol)
    sentiment_results = sentiment_agent.analyze_social_media_sentiment(posts)
    signals = []
    for result in sentiment_results:
        signal = strategy_agent.generate_trading_signal(result["score"])
        signals.append(signal)
    sentiment_df = pd.DataFrame(sentiment_results)
    sentiment_df["Signal"] = signals
    st.write("Social Media Sentiment Analysis:")
    st.dataframe(sentiment_df)
    st.bar_chart(sentiment_df.set_index("post")["score"])

if st.button("Backtest Strategy"):
    if "Signal" in data.columns:
        final_capital = strategy_agent.backtest_strategy(data)
        st.write(f"Final capital after backtesting: ${final_capital:.2f}")
    else:
        st.write("No trading signals available for backtesting.")

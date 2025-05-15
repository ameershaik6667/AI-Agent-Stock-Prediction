# AI-Agent-Stock-Prediction

This code is from Rivier University COMP-699 Professional Seminar student projects.

They create a classical trading system (e.g., 50/200 SMA cross) and then enhance it with multiple AI agents.

They backtest the trading system using AI agents against the classical system.

## Executing the Code


```sh
(stocks) jglossner@jglossner:~/GitRepos/AI-Agent-Stock-Prediction$ streamlit run src/UI/app.py
```
or
```sh
(stocks) jglossner@jglossner:~/GitRepos/AI-Agent-Stock-Prediction$ python -m src.UI.gap
```


## Recommended Installation

Use a github codespace


## Local Installation

### Install Anaconda

Install [Anaconda Python](https://www.anaconda.com/download).

or an alternative for Linux:

```sh
wget --no-check-certificate https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh -O /tmp/anaconda.sh && \
    sudo bash /tmp/anaconda.sh -b -p /opt/conda && \
    rm /tmp/anaconda.sh
export PATH="/opt/conda/bin:$PATH"
```

### Create conda environment
```sh
conda env create -f conda_env.yml
conda init
conda activate stocks
```

## Set Environment Variables

Sign up to get an [OpenAI Key](https://platform.openai.com/docs/overview)
Sign up to get a [free SEC API Key](https://sec-api.io/)
Sign up to get a [free SERPER API Key](https://serper.dev/)

```sh
export OPENAI_API_KEY=sk-     # available form platform.openai.com
export SEC_API_API_KEY= your long list of numbers   # Sign up for a free key
export SERPER_API_KEY= your key # Free for 2500 queries
```
Note: for Windows use *set* instead of *export*

## Set up Selenium and the Chromium webdriver

Download the [chromedriver](https://googlechromelabs.github.io/chrome-for-testing/#stable) from the stable channel.

Place it is a folder named chromedriver in the root directory. This will not be on github because some students need Linux or MAC versions.

Note that it must match the version of Chrome on your computer. You can check it by starting the Chrome browser. Then navigate to on your browser to the top right 3 dots, help->About Chrome. 

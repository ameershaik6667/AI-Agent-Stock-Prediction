# AI-Agent-Stock-Prediction

This code is from Rivier University COMP-699 Professional Seminar student projects.

They create a classical trading system (e.g., 50/200 SMA cross) and then enhance it with multiple AI agents.

They backtest the trading system using AI agents against the classical system.

## Executing the Code


```sh
~/AI-Agent-Stock-Prediction/src/Backtesting$ uv run streamlit run backtest_adx.py
```
or
```sh
~/AI-Agent-Stock-Prediction$ uv run python -m src.UI.gap
```


## Recommended Installation

Use a github codespace


## 

## Local Windows

### Install WSL and Ubuntu24.04

https://learn.microsoft.com/en-us/windows/wsl/install

open PowerShell in admin mode (right click on program)
`wsl --install -d Ubuntu-24.04`

To see all the available Linux distributions

`wsl --list --online`

reboot your machine



### Install Docker Desktop

https://docs.docker.com/desktop/setup/install/windows-install/

reboot 

start Docker desktop and configure it to start on Windows boot (Settings->General)



### Open Ubuntu in WSL

In Windows search, type Ubuntu and select Ubuntu-24.04

create your userid

create a password  <---- DON'T FORGET IT



### Follow the Install Linux Software Instructions

From here, the directions for Linux and Windows running Linux are the same except where noted.



# Install Linux Software (in Ubuntu or WSL Ubuntu)

`sudo apt update && sudo apt install -y \
    software-properties-common \
    curl \
    zip \
    unzip \
    tar \
    ca-certificates \
    git \
    wget \
    build-essential \
    vim \
    jq \
    firefox \  
    wslu \
    && sudo apt clean`



### Install uv and venv

https://docs.astral.sh/uv/#installation

`curl -LsSf https://astral.sh/uv/install.sh | sh`



### Install Microsoft Visual Studio Code

https://code.visualstudio.com/sha/download



### Clone the Repository

`git clone https://github.com/Rivier-Computer-Science/AI-Agent-Stock-Prediction.git`

cd into Adaptive-Learning and initialize a venv environment

`uv venv --python 3.12`

Activate the environment

`source .venv/bin/activate`



### Install Python requirements.txt

`uv pip install -r requirements.txt`



### Set up the Default Browser for Windows Display

Note: Linux users should not need to perform this step.

If your Windows browser does not open automatically:

Option 1: All http requests use the Windows browser:
`sudo apt install wslu
echo 'export BROWSER=wsluview' >> ~/.bashrc`

Option 2: Only this project uses the Windows browser:

`sudo apt install wslu`

echo 'export BOKEH_BROWSER=wsluview' >> ~/.bashrc`

Option 3: Run the browser from within Ubuntu: :
`echo 'export BOKEH_BROWSER=firefox' >> ~/.bashrc`



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


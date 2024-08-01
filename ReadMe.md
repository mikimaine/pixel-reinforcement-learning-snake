# Project 1: Reinforcement Learning Model with Bellman Equation

## Project setup

The project was developed using Python 3.11, PyGame, Tensorflow, and other python libraries.

### Ubuntu

If using Ubuntu you might need some packages:

```bash
sudo apt-get install -y \
    qttools5-dev-tools \ 
    libfreetype6-dev libpng-dev \
    libjpeg-dev zlib1g-dev libtiff-dev liblcms2-dev libwebp-dev \
    libffi-dev python3-dev build-essential \
    libpq-dev libmysqlclient-dev \
    libatlas-base-dev gfortran
```

### Conda
If using `conda`, run the following command to create the environment : 

```bash
conda env create -f environment.yml
```

To activate the environment use:
```bash
conda activate pixel
```

### venv
If using `venv`, run the following commands to set up the environment:

```bash
python -m venv venv
source venv/bin/activate
```

### Python packages installation

To install the python packages used:
```bash
pip install -r requirements.txt
```

## Running the project

To run the project, execute:

```bash
python main.py
```

A window with start up settings will be presented. To run the game driven by the agent, click `Train Agent`.

![Settings Window](training.png)

## Main Assets
    - Enviroment
    - Game 
    - Agent
    - Model
    - Setting

## Define each Assets

1. Environment
    - Python > 3.10
    - venv
    - pygame
    - numpy
    - Pytorch

2. Game
    - Action
    - Point 
    - game loop
    - move
    - food
    - game over
    - reset

3. Agent
    - state
    - memory
        - short memory
        - long memory
    - action
    - train
    - reward ( +10 for score, -10 for die)

4. Model
    - QNet  [Linear torch ref](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
    - losses MeanSquare
    - QLearning [Bellman DNQ algorithm] (https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
    - Save and load model

5. Setting
    - batch
    - learning rate
    - gamma
    - activation function
    - optimizer/loss function
    - game speed
    - reward amount
    - hidden layer size, number of neuron size

6. Reporting
    - Plot score and number of game
    - Real time progress
    - Amount of reward / win

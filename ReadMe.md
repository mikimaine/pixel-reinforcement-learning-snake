# Project 1: Reinforcement Learning Model with Bellman Equation

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
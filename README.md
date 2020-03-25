# COMP3931-model
Final year project: computational economics, agent-based model, financial accelerator

## Introduction

## Usage


## Model

Our economy is built with two markets: consumer market and credit market.
Consumers buy from firms on the consumer market and firms buy credit from banks on
the credit market.

Our credit is very basic, we assume that consumers buy all firms output.
Where each firm sells goods at a stochastic price, following a normal distribution.
We use "dymanic trade-off theory" by letting firms follow an adaptice behavioral rule
for leverage, ![L_{i,t}](https://render.githubusercontent.com/render/math?math=L_%7Bi%2Ct%7D).

The amount of debt firms can borrow from banks is determined by 
![B_{i,t}=A_{i,t}L_{i,t}](https://render.githubusercontent.com/render/math?math=B_%7Bi%2Ct%7D%3DA_%7Bi%2Ct%7DL_%7Bi%2Ct%7D)

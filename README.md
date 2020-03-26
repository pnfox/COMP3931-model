# COMP3931-model

## Introduction

Agent-based simulation of a financial accelerator.

This project provides the underlining model and tools to perform easy analysis.

This work is originally based off [Riccetti et al (2013)](https://www.sciencedirect.com/science/article/pii/S0165188913000419)

## Model

Our simulated economy is initially setup with a number of firms, banks, steps to run and
other parameters.

Firms sell goods that they make and buy credit from banks to keep production going.
We assume that consumers buy all firms output.
The amount of credit that a firm can borrow depends on its previous performance, we use
"dymanic trade-off theory" by letting firms follow an **adaptive behavioral rule
for leverage**.

## Usage

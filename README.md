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

Once this repository has been cloned locally run the following command:

```pip install -r requirements.txt```

This will download and install all python modules required to run this project.

Note: Python >= 3.5 is required, use `virtualenv` to create a python enviroment
with an appropiate version by running `virtualenv -p PYTHON_EXE DEST_DIR` to use
to the correct python version.

Running a simulation is done via the `main.py` script.

```
usage: main.py [-h] -t SIMULATIONSTEPS -f FIRMS -b BANKS [-alpha PRICEMEAN]
               [-var PRICEVARIANCE] [-gamma GAMMA] [-partners CHI]
               [-intensity CHOICEINTENSITY] [-adjustment ADJUSTMENT]
               [-phi PHI] [-beta BETA] [-rCB RCB] [-bankcost BANKCOST] [-i]
               [-s SEED] [-o FOLDER]

Economic ABM model simulation

optional arguments:
  -h, --help            show this help message and exit
  -t SIMULATIONSTEPS, --timesteps SIMULATIONSTEPS
                        Steps simulation is ran for
  -f FIRMS, --firms FIRMS
                        Number of Firms in Simulation
  -b BANKS, --banks BANKS
                        Number of Banks in Simulation
  -alpha PRICEMEAN      firm price mean
  -var PRICEVARIANCE    firm price variance
  -gamma GAMMA          interest rate parameter
  -partners CHI         number of potential partners on credit market
  -intensity CHOICEINTENSITY
                        Intensity of partner choice
  -adjustment ADJUSTMENT
                        Leverage adjustment
  -phi PHI              Firm production function parameter
  -beta BETA            Firm production function parameter
  -rCB RCB              Central bank interest rate
  -bankcost BANKCOST    Cost of maintaining banks
  -i, --interactive     interactive mode - Useful for debugging
  -s SEED, --seed SEED  Simulation seed
  -o FOLDER, --output FOLDER
                        Directory location where results are written
```

### Dependencies

* [Numpy](https://numpy.org)

* [Scipy](https://www.scipy.org)

* [Matplotlib](https://matplotlib.org/)

### License

This project is licensed under the GNU GPLv3 Lincense - see [LICENSE.md](LICENSE.md)

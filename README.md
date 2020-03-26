# COMP3931-model
Final year project: computational economics, agent-based model, financial accelerator

## Introduction

## Usage


## Model

Our simulated economy is initially setup with a number of firms, banks, steps to run and
other parameters.

#### Firm capital structure

Firms sell goods that they make and buy credit from banks to keep production going.
We assume that consumers buy all firms output.
Each firm sells goods at a stochastic price, following a normal distribution.
The amount of credit that a firm can borrow depends on its previous performance, we use
"dymanic trade-off theory" by letting firms follow an **adaptive behavioral rule
for leverage**, ![L_{i,t}](https://render.githubusercontent.com/render/math?math=L_%7Bi%2Ct%7D).

![L_{i,t} = \begin{Bmatrix} L_{i,t}(1+adj*u) & p_{i,t} > R_{i,z,t}\\ L_{i,t}(1-adj*u) & p_{i,t} \leq R_{i,z,t} \end{Bmatrix}](https://render.githubusercontent.com/render/math?math=L_%7Bi%2Ct%7D%20%3D%20%5Cbegin%7BBmatrix%7D%20L_%7Bi%2Ct%7D(1%2Badj*u)%20%26%20p_%7Bi%2Ct%7D%20%3E%20R_%7Bi%2Cz%2Ct%7D%5C%5C%20L_%7Bi%2Ct%7D(1-adj*u)%20%26%20p_%7Bi%2Ct%7D%20%5Cleq%20R_%7Bi%2Cz%2Ct%7D%20%5Cend%7BBmatrix%7D)

Where ![R_{i,z,t}](https://render.githubusercontent.com/render/math?math=R_%7Bi%2Cz%2Ct%7D) is
the amount of interest rate given to firm ![i](https://render.githubusercontent.com/render/math?math=i)
from bank ![z](https://render.githubusercontent.com/render/math?math=z) at time ![t](https://render.githubusercontent.com/render/math?math=t) See next section for more.
![adj](https://render.githubusercontent.com/render/math?math=adj) sets the maximum leverage change
and ![u](https://render.githubusercontent.com/render/math?math=u) follows a uniform distribution between 0 and 1. Firm price follow a normal distribution ![p_{i,t}\sim \mathcal{N}(\alpha, var)](https://render.githubusercontent.com/render/math?math=p_%7Bi%2Ct%7D%5Csim%20%5Cmathcal%7BN%7D(%5Calpha%2C%20var))

The amount of debt firms can borrow from banks is determined by 
![B_{i,t}=A_{i,t}L_{i,t}](https://render.githubusercontent.com/render/math?math=B_%7Bi%2Ct%7D%3DA_%7Bi%2Ct%7DL_%7Bi%2Ct%7D)
Meaning that firms will borrow more credit if ![p_{i,t} > R_{i,z,t}](https://render.githubusercontent.com/render/math?math=p_%7Bi%2Ct%7D%20%3E%20R_%7Bi%2Cz%2Ct%7D), otherwise they ask for less credit.
where ![A_{i,t}](https://render.githubusercontent.com/render/math?math=A_%7Bi%2Ct%7D) is the firms networth, described by ![A_{i,t+1} = A_{i,t} + Pr_{i,t}](https://render.githubusercontent.com/render/math?math=A_%7Bi%2Ct%2B1%7D%20%3D%20A_%7Bi%2Ct%7D%20%2B%20Pr_%7Bi%2Ct%7D)

Firm profits are computed as follows:

![Pr_{i,t}=p_{i,t}Y_{i,t}-R_{i,t}B_{i,t}](https://render.githubusercontent.com/render/math?math=Pr_%7Bi%2Ct%7D%3Dp_%7Bi%2Ct%7DY_%7Bi%2Ct%7D-R_%7Bi%2Ct%7DB_%7Bi%2Ct%7D)

where ![Y_{i,t}](https://render.githubusercontent.com/render/math?math=Y_%7Bi%2Ct%7D) is firm
output decribed below.

A firms capital is then defined as its networth plus its debt. ![K_{i,t}=A_{i,t}+B_{i,t}](https://render.githubusercontent.com/render/math?math=K_%7Bi%2Ct%7D%3DA_%7Bi%2Ct%7D%2BB_%7Bi%2Ct%7D)

How much goods firms produce, their output ![Y_{i,t}](https://render.githubusercontent.com/render/math?math=Y_%7Bi%2Ct%7D) is described by

![Y_{i,t}=\phi K_{i,t}^{\beta}](https://render.githubusercontent.com/render/math?math=Y_%7Bi%2Ct%7D%3D%5Cphi%20K_%7Bi%2Ct%7D%5E%7B%5Cbeta%7D)


#### The way firm and bank interact

In each step a firm asks a bank for debt as described above. Each bank sets an interest rate
which applies to all its customers, but note that firms can only interact with one bank at
a time. Every period each firm observes the interest rates of ![\chi](https://render.githubusercontent.com/render/math?math=%5Cchi)
randomly selected banks and a better interest rate (i.e lower) than its current banks then
it will switch partners.

Interest rates follow the rule:

![R_{i,z,t}=c(A_{z,t}/B_{z,t})+(c+r^{CB})(D_{z,t}/B_{z,t})+\theta +\eta L_{i,t}BAD_{t-1}/B_{t-1}](https://render.githubusercontent.com/render/math?math=R_%7Bi%2Cz%2Ct%7D%3Dc(A_%7Bz%2Ct%7D%2FB_%7Bz%2Ct%7D)%2B(c%2Br%5E%7BCB%7D)(D_%7Bz%2Ct%7D%2FB_%7Bz%2Ct%7D)%2B%5Ctheta%20%2B%5Ceta%20L_%7Bi%2Ct%7DBAD_%7Bt-1%7D%2FB_%7Bt-1%7D)

Where ![BAD_{t-1}](https://render.githubusercontent.com/render/math?math=BAD_%7Bt-1%7D) is the
aggregate bad debt and ![D_{z,t}](https://render.githubusercontent.com/render/math?math=D_%7Bz%2Ct%7D)
is the banks deposits, computed as the sum of all loans.

Therefore a banks profits are computed as follows:

![Pr_{z,t}=\sum_{i}R_{i,z,t}B_{i,z,t}+r^{CB}_{t}D_{z,t}-c(A_{z,t}+D_{z,t})-bad_{z,t}](https://render.githubusercontent.com/render/math?math=Pr_%7Bz%2Ct%7D%3D%5Csum_%7Bi%7DR_%7Bi%2Cz%2Ct%7DB_%7Bi%2Cz%2Ct%7D%2Br%5E%7BCB%7D_%7Bt%7DD_%7Bz%2Ct%7D-c(A_%7Bz%2Ct%7D%2BD_%7Bz%2Ct%7D)-bad_%7Bz%2Ct%7D)

where ![bad_{z,t}](https://render.githubusercontent.com/render/math?math=bad_%7Bz%2Ct%7D) is
banks bad debt, the sum of all credit lent to firms defaulted in period t.



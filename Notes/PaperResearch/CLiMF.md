# [CLiMF: Collaborative Less-is-More Filtering](https://www.ijcai.org/Proceedings/13/Papers/460.pdf)

## Overview

> less-is-more: provide users with only few but valuable recommendations

Feature

* focused on improving top-k recommendation (making a few but relevant recommenddations)
* is tailored to recommendation domains where only *binary relevance data* is available

Approach

* Directly maximizing the Mean Reciprocal Rank (MRR)

Dataset
  
* [Epinions](http://www.trustlet.org/epinions.html)
* Tuenti

### Reciprocal Rank (RR)

> In this paper, they introduce a way to significantly reducing the computational complexity of RR optimization

## CLiMF Algorithm

### Smoothing the RR

### Lower Bound of Smooth RR

objective funciton of CLiMF

$$
F(U, V) = \sum^M_{i=1}\sum^N_{j=1}Y_{ij}[\ln g(U_i^T V_j) + \sum^N_{k=1}\ln(1-Y_{ik}g(U_i^T V_k - U_i^T V_j))] - \frac{\lambda}{2}(||U||^2 + ||V||^2)
$$

### Optimization

Use *stochastic gradient ascent* to maximize the deriviatives of objective function with respect to $U_i$ and $V_j$

## Resources

### Source Code

* [gamboviol/climf](https://github.com/gamboviol/climf)
* [coreylynch/pyCLiMF](https://github.com/coreylynch/pyCLiMF)

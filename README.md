# Using reinforcement learning in conformal and modular bootstrap

Solving for the spectrum and OPE coefficients in conformal bootstrap amounts to minimizing the penalty for not satisfying crossing equations. Recently, an attempt was made in [1] to use reinforcement learning technique in machine learning to perform this minimization. The idea is that, with a particular set of external operators and spin assignment for exchange operators in a given window, one can iteratively modify the conformal dimensions and the OPE coefficients to stay close to the crossing plane as close as possible. Compared to the usual minimization program, the authors in [1] claimed that RL can learn the structure of the problem and can thus deal with the curse of dimensionality more easily as we increase the number of operators(unknowns).

We implement our models as open.ai gym environments[2]. Currently available environments:
* Four spin correlator in 2D Ising



# Reference

1. arXiv:2108.09330v2
3. https://gym.openai.com/
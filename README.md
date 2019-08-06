# Bootstrapped policy gradient for difficutly adaptation
This is implementation for aamas'19 paper of ["Bootstrapped Policy Gradient for Difficulty Adaptation in Intelligent Tutoring Systems"](http://www.ifaamas.org/Proceedings/aamas2019/pdfs/p711.pdf).

![](http://latex.codecogs.com/gif.latex?\\frac{\\partial J}{\\partial \\theta_k^{(j)}}=\\sum_{i:r(i,j)=1}{\\big((\\theta^{(j)})^Tx^{(i)}-y^{(i,j)}\\big)x_k^{(i)}}+\\lambda \\xtheta_k^{(j)})

The folder of "Difficulty Adaptation" contains the implementation of BPG for difficultu adaptation using simulation data.
To run the code using the following command:

`python main.py`

The folder of "continuous_bandit_bpg" contains the implementation of BPG for continuous-armed bandit.
To run the code using the following command:

`python Continuous_Bandit_main_OO.py`



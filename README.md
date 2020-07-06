# Bootstrapped policy gradient for difficulty adaptation
This is implementation for aamas'19 paper of ["Bootstrapped Policy Gradient for Difficulty Adaptation in Intelligent Tutoring Systems"](http://www.ifaamas.org/Proceedings/aamas2019/pdfs/p711.pdf).
More information on BPG project can be found at [here](https://yaqianzhang.github.io/2018/06/30/boostrapped-policy-gradient.html) and the application of BPG on a visual memory game can be found at [here](https://yaqianzhang.github.io/2018/06/30/difficulty-adjustment-for-visual-memory-training.html)
## What is Bootstrapped Policy Gradient (BPG)?
The key idea is to improve the sample efficiency by updating the probability of _a set of actions_ instead of a single action in the gradient sample.
We propose to a surrogate policy gradient direction to encourage _better actions_ and discourage _worse actions_.
## Key advantages of BPG
BPG can achieve fast and stable convergence with small batch size (even batch size of 1). This makes it suitable for environments with large action space and short exploration horizon.

## Source Code
The folder of "Difficulty Adaptation" contains the implementation of BPG for difficulty adaptation using simulation data.
To run the code using the following command:

`python main.py`

The folder of "continuous_bandit_bpg" contains the implementation of BPG for continuous-armed bandit.
To run the code using the following command:

`python Continuous_Bandit_main_OO.py`

## Dependencies：
Python3
numpy




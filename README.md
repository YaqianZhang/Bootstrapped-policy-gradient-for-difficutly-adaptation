# Bootstrapped policy gradient for difficutly adaptation
This is implementation for aamas'19 paper of ["Bootstrapped Policy Gradient for Difficulty Adaptation in Intelligent Tutoring Systems"](http://www.ifaamas.org/Proceedings/aamas2019/pdfs/p711.pdf).

![](http://latex.codecogs.com/gif.latex?\\\begin{equation}
\label{eq_bpg}
\begin{aligned}
{\tilde{\nabla }_{\theta }}{J}(\theta )={{\mathbb{E} }_{{{a}_{i}}{\sim}{{\pi }_{\theta }}}}[{|{r}_{{{a}_{i}}}|}({{\nabla }_{\theta }}\log {{{\overset{\scriptscriptstyle\frown}{\pi }}}^+_{\theta }}({{a}_{i}})
-{{\nabla }_{\theta }}\log {{{\overset{\scriptscriptstyle\frown}{\pi }}}^-_{\theta }}({{a}_{i}}))]   
\end{aligned}
\end{equation})

The folder of "Difficulty Adaptation" contains the implementation of BPG for difficultu adaptation using simulation data.
To run the code using the following command:

`python main.py`

The folder of "continuous_bandit_bpg" contains the implementation of BPG for continuous-armed bandit.
To run the code using the following command:

`python Continuous_Bandit_main_OO.py`



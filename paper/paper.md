---
title: 'Abmarl: Connecting Agent-Based Simulations with MultiAgent Reinforcement Learning'
tags:
  - Python
  - agent-based simulation
  - multiagent reinforcement learning
  - machine learning
  - agent-based modeling
authors:
  - name: Edward Rusu
    orcid: 0000-0003-1033-439X
    affiliation: 1
  - name: Ruben Glatt
    affiliation: 1
affiliations:
 - name: Lawrence Livermore National Laboratory
date: 9 June 2021 # TODO: Update this!
bibliography: paper.bib
---

# Summary

Abmarl is a package for developing Agent-Based Simulations and training them
with MultiAgent Reinforcement Learning (MARL). We provide an intuitive command line
interface for engaging with the full workflow of MARL experimentation: training,
visualizing, and analyzing agent behavior. We define an Agent-Based Simulation
Interface and Simulation Manager, which control which agents interact with the
simulation at each step. We support integration with popular reinforcement learning
simulation interfaces, including gym.Env `[@gym]` and MultiAgentEnv `[@rllib]`.
We leverage RLlib's framework for reinforcement learning and extend it to more easily
support custom simulations, algorithms, and policies. We enable researchers to
rapidly prototype MARL experiments and simulation design and lower the barrier
for pre-existing projects to prototype RL as a potential solution.

# Statement of need

In 2016, `@gym` published OpenAi Gym, an interface for single-agent simulations. This interface
has become one of the most popular connections between simulation and training in reinformcent learning
experimentation. It has been used by the standard set of benchmarks for single-agent
reinforcement learning, including the Arcade Learning Environment `[@arcade]`.
Since then the field of DRL has exploded in both algorithm development
and simulation design, and over the past a few years researchers have been extending
their interest to MultiAgent Reinforcement Learning (MARL).

MARL has shown exceptional promise towards artificial
general intelligence. Surprisingly complex and hierarchical behavior emerges in the
interaction among multiple agents, especially when those agents differ in their
objectives `[@hide-n-seek]`....

... Several projects have attempted to define a standard set
of benchmark scenarios for MultiAgent problems; such as MAgent `[@magent]`, Starcraft `[@smac]`, and
Neural MMO `[@neuralmmo]`. However, each of these couples the interface with the 
underlying simulation. Notably, `@pettingzoo` has attempted to unify some of
the more popular simulations under a single interface, giving researchers easier
and access to these simulations. While this is a step towards
a standard multiagent interace, these are still tied to a specific set of already-built simulations.

Abmarl provides 

Abmarl appr

...
The field of MARL provides an overwhelming set of choices for
the simulations, the learning algorithms, and the software frameworks. Among these,
RLlib `[@rllib]` has risen as one of the top choices, supporting many state-of-the-art
algorithms, support for training on HPC platforms, and an open-ended simulation interface.
Although RLlib provides the skeleton for connecting simulations with trainers, its
simulation interface does not provide 
While their most popular `MultiAgentEnv`.




Running MultiAgent Reinforcement Learning experiments 

<!-- TODO: Supply a statement of need of connecting Agent Based Simulations with MultiAgent
Reinforcement Learning.
Main points:
1. Gym.Env is a standard interfance for single-agent RL that allowed practitioners
    to set a speciic goal towards an simulation design, reuse other's work, create
    algorithms designed for a single simulation interface.
2. Movement towards MARL, and while there have been lots of great baseline simulatoin
    (include citations here), they have all been tied to specific simulation.
3. Abmarl provides an interface for simulations that are naturally implemented
    as agent-based simulations.
4. Additionally, Abmarl simplifies the "learning curve" needed to get started with
    MARL libraries, such as RLlib, lowering the barrier for researchers who are
    interested in usinng RL in their work.
-->

# Highlights

<!-- Brief section highlighting Abmarl's use in plume and emergent adversarial behaviors. -->

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References

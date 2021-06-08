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
bibliography: paper/paper.bib
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
defined one of the most popular connections between simulation and training in reinformcent learning
experimentation. It has been used by many simulation benchmarks for single-agent
reinforcement learning, including the Arcade Learning Environment `[@arcade]`.
Since then the field of DRL has exploded in both algorithm development
and simulation design, and over the past a few years researchers have been extending
their interest to MultiAgent Reinforcement Learning (MARL).

MARL has shown exceptional promise towards artificial
general intelligence. Surprisingly complex and hierarchical behavior emerges in the
interaction among multiple agents, especially when those agents differ in their
objectives `[@hide-n-seek]`. Several projects have attempted to define a standard set
of benchmark scenarios for MultiAgent problems; such as MAgent `[@magent]`, Starcraft `[@smac]`, and
Neural MMO `[@neuralmmo]`. However, each of these couples the interface with the 
underlying simulation. Notably, `@pettingzoo` has attempted to unify some of
the more popular simulations under a single interface, giving researchers easier
and access to these simulations. While this is a step towards
a standard multiagent interace, these are still tied to a specific set of already-built simulations
with limited flexibility.

Abmarl defines a generalized interface for multiagent simulations that is versatile,
generalizable, extendible, and intuitive. Rather than adapting gym's interface for a targetted
multiagent simulation, we have built an interface from scratch that allows for the greatest flexbility
while still connecting to the top open-source RL library, namely RLlib. Our interface
manages the loop between agents and the trainer, enabling the researcher to focus
on simulation design or algorithmic development without worrying about the data exchange.

Finally, Abmarl's intuitive command-line interface gives researchers a running-start
in MARL experimentation. We handle all the workflow elements needed to setup, run,
and reproduce MARL experiments, providing direct abilities to train, visualize,
and anaylze experiments. We streamline the savy-practictioners experience and lower
the barrier for new researchers to join the field.

# Highlights

Abmarl has been used in the following research efforts:

1. Abmarl has been used in Hybrid Information-driven Multi-agent Reinforcement Learning
`[@hybrid]`, where multiple agents work together to construct a posterior distribution of a
chemical plume's source location. Each agent is equipped with a set of discrete
actions that are heuristically implemented, and the high-level choosing of each action is
trained using MARL. The simulation was setup using Abmarl's Simulation interface
and the training was mananged via our command line interface.
2. Abmarl has been used to study emergent behaviors in scnearios involving teams
of heterogeneous agents, where each teammate shares the same objective. The
competition among teams results in collaboration among team members and competition
among the teams, all from a sparse reward schema. This is ongoing research.
3. Abmarl will also provide a cirriculum of tasks in MARL scenarios to serve as
a benchmark for new algorithms. The framework for this is open-ended and user-friendly,
allowing researchers to easily define a virtually inifinte set of grid-based simulations
for training. This is ongoing development.

<!-- # Mathematics

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

-->
# Acknowledgements

This work was performed under the auspices of the U.S. Department of Energy by
Lawrence Livermore National Laboratory under contract DE-AC52-07NA27344. Lawrence 
Livermore National Security, LLC through the support of LDRD 20-SI-005, 20-COMP-015,
and 21-COMP-017. LLNL-CODE-815883.

# References

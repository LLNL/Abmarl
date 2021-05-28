.. Admiral documentation overview.

Design
======

A reinforcement learning experiment contains two main components: (1) a simulation
environment and (2) learning agents, which contain policies that map observations
to actions. These policies may be hard-coded by the researcher or trained
by the RL algorithm. In Admiral, these two components are specified in a single
Python configuration script. The components can be defined in-script or imported
as modules.

Once these components are setup, they are passed as parameters to RLlib's
tune command, which will launch the RLlib application and begin the training
process. The training process will save checkpoints to an output directory,
from which you can visualize and analyze results. The following diagram
demonstrates this workflow.

![Workflow](.images/workflow.png)

#TODO: more content here, especially talking about the AES (agent-environment simulation)/
ABS nature of the repository and environments.

Creating Agents and Environments
--------------------------------

Using Agent, AgentBasedSimulation, and Managers to construct a simulation ready
for training in Admiral.

Experiment Configuration
------------------------
We must define a configuration script that specifies the environment and agent
parameters. Once we have this script, we can use the command-line interface
to train, visualize, and analyze agent behavior.


Visualizing
-----------

Analyzing
---------



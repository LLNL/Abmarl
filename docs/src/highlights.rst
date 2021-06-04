.. Admiral documentation highlights.

Admiral Highlights
==================


Emergent Collaborative and Competitive Behavior
-----------------------------------------------

In this experiment, we study how collaborative and competitive behaviors emerge
among agents in a partially observable stochastic game. In our simulation, each
agent occupies some grid square and can move around the map. Each agent can attack
agents that are on a different "team"; the attacked agent loses its life and
is removed from the simulation. Each agent can observe the state of the map in
a region surrounding its location. It can see other agents and what team they're
on as well as the edges of the map. The diagram below visuially depicts the agents'
observation and action spaces.

.. image:: .images/grid_agent_diagram.png
   :width: 100 %
   :alt: Diagram visually depicting agents' observation and action spaces.

In the various examples below, each policy is a two-layer MLP, with 64 unites in
each layer. We use RLlib's A2C Trainer with default parameters and train for
two million episodes on a compute node with 72 CPUs, a process that takes 3-4
hours per experiment.

Single Agent Foraging
`````````````````````
We start by considering a single foraging agent whose objective is to move around
the map collecting resources. The resources are technically agents themselves,
although they don't do anything and don't learn anything. The single foraging agent
can see up to 3 squares away and can move up to 1 square away. The agent is rewarded
for every resource it collects and given a small penalty for attempting to move
off the map and an even smaller "entropy" penalty every time-step to encourage
it to act quickly. At the beginning of every episode, the agent and resources spawn
at random locations in the map. Below is a video showing a typical full episode
of the learned behavior and a brief analysis.

.. figure:: .images/single_agent_full.gif
   :width: 100 %
   :alt: Video showing an episode with the trained behavior.

   A full episode showing what the forager learned. The forager is the blue circle
   and the resources are the green squares. Notice how the forager bounces among
   resource clusters, collecting all local resources before exploring the map for
   more.

When it can see resources
'''''''''''''''''''''''''
The agent moves toward the closest resource that it observes and forages it. Note
that the agent's foraging range is 1 square away from itself: the agent rarely
waits until it is directly over the resource before foraging it; it usually foraging
as soon as it is within range. In some cases, the agent intelligently places itself
in the middle of 2-3 resources in order to forage within the least number of moves.
When the resources are near the edge of the map, the agent behaves with some inefficiency,
likely due to the small penalty we give it for moving off the map, which results
in an aversion towards the map edges. Below is a series of short video
clips showing how the agent behaves when it can see resources.

.. figure:: .images/single_agent_exploit.gif
   :width: 100 %
   :alt: Video showing the forager's behavior when it observes resources.

   The forager learns an effective foraging strategy, moving towards and foraging
   the nearest resources that it observes.

When it cannot see resources
'''''''''''''''''''''''''''''
The foragers behavior when it is near resources is not surprising. But how does
the forager behaves when it cannot see any resources? The forager only sees that
which is near it and does not have any information distinguishing one "deserted"
area of the map from another. Recall, however, that the agent observe the edges
of the map, and it uses this information to learn an effecive exploration policy.
In the video below, we can see that the agent learns to explore the map by moving
along its edges in a clockwise-direction, occasionally making random moves towards
the middle of the map.

.. figure:: .images/single_agent_explore.gif
   :width: 100 %
   :alt: Video showing the forager's behavior when it does not observe resources.

   The forager learns an effective exploration policy: it moves along the edge
   of the map in a clockwise direction.

.. IMPORTANT::
   We do not use any kind of heuristic or mixed policy. The exporation is behavior
   that *emerges* entirely from reinforcement learning.

Multiple Agents Foraging
````````````````````````
Having experimented with what a single forager can learn, we turn our attention
to the behaviors learned by multiple foragers interacting in the map at the same
time. Each forager is homogeneous with each other as described above: they can
all move up to one square away, observe up to three squares away, and are rewarded
the same way. The observations include other foragers in addition to the resources
and map edges. All agents train a single policy. Below is a brief analysis of the
learned behaviors.

Cover and explore
'''''''''''''''''
Our reward schema implicity encourages the agents to collaborate because we give
a small penalty to each agent for taking too long. Thus, the faster the agents
can collect all the resources, the less they are penalized. Furthermore, because each
agent trains the same policy, there is no incentive for competitive behavior. An
agent can afford to say, "I don't need to get the resource first. As long as one
of us gets it quickly, then we all benefit". Therefore, the foragers learn to spread
out to *cover* the space.

In the video clips below, we see that the foragers avoid being within observation
distance of one another. Typically, when two foragers get too close, they repel
each other, each moving in opposite directions, ensuring that the space is *covered*.
Furthermore, notice the dance-like exploration policy that these foragers learn.
Similar to the single-agent case above, the foragers learn to *explore* along the
edges of the map in a clockwise direction. However, they're not as efficient as
the single agent because they "repel" each other.

.. figure:: .images/multi_agent_spread.gif
   :width: 100 %
   :alt: Video showing how the foragers spread out.

   The foragers cover the map by spreading out and explore it by traveling in a
   clockwise direction.

.. IMPORTANT::
   We do not directly incentivize agents to keep their distance. No part of the
   reward schema directly deals with the agents' distances from each other. These
   learned behaviors are *emergent*.

Breaking the pattern
''''''''''''''''''''
When a forager observes a resource, it breaks its "cover and explore" behavior and
moves directly for the resource. Even multiple foragers move towards the same resource.
They have no reason to coordinate who will get it because, as we stated above,
there is no incentive for competition, so no need to negotiate. If another forager
gets there first, everyone benefits. The foragers learn to prioritize collecting
the resources over keeping their distance from each other.

.. figure:: .images/multi_agent_forage.gif
   :width: 100 %
   :alt: Video showing how the foragers move towards resources.

   The foragers move towards resources to forage, even when there are other foragers
   nearby.

.. Tip::
   We should expect to see a mixture of these "behavioral regimes" occuring at
   the same time within a simulation because while some agents are "covering and
   exploring", others are moving towards resources.

Introducing Hunters
```````````````````


Hunters effectively hunt agents.
Highlight some of the behaviors I put in the slides.



## DQN: Where it fails and how to prevent this

https://pdfs.semanticscholar.org/7713/18e96c7df721135d92a78740f5ebf8696909.pdf



In reinforcement learning the agent needs to learn which actions to select in unknown, dynamic environments such that it will achieve its goal or maximise its objective function \cite{sutton2011reinforcement}. There are a few methods which help us achieve this goal. First, there is Tabular Methods. This method calculates a so called Q-value for each state. However, in many tasks the state space (space where each state is represented) is enormous. This makes it almost impossible to visit each state and determine the Q-value because this simply takes too much time and requires too much data. This is why, in case of an enormous state space, we are forced to find an approximation of the Q-values. This is done by estimating a function that determines the Q-value for each state (i.e. function approximation). 

With function approximation, you look at the difference between the target value (value that Q should be) and the current value that our function produces. This current value is then updated based on the target value. 

## Choosing the target

There are a few different ways to chose this target. First of all, we could use a method called Monte Carlo. With Monte Carlo, multiple episodes are traversed and the value of Q is sampled from these episodes. We could also decide to use a method that uses bootstrapping to determine the target. 

Methods that use bootstrapping have certain adventages over Monte Carlo methods. First of all, bootstrapping methods allow for online learning: learning while training. Monte Carlo must go through the entire episode in order to update the Q-value. Bootstrapping methods are able to do updates while an epsidoe is ongoing. This online learning means that an agent is less likely to get stuck in an infinite episode. 


A disadvantage is that using a bootstrapping method results in a semi-gradient. \todo{uitleggen dat gradietns te maken hebben met leren?} Usually, a semi-gradient will still converge (meaning that an optimal value function will be found). However, bootstrapping methods can sometimes diverge. This is the case if one used off-policy learning while also making use of a behaviour policy. Divergence here means that no optimal solution will be found. 

The rist of divergence is highest if all of the following three elements are present: 

* Function approximation
* Bootstrapping
* Off-policy training

These three elemetns are often called "the deathly triad". 


Luckily, a few researches found a version of DQN \todo{deze term eerder introduceren} which does not diverge. They tackle the problem by introducing experience replay. Experience relay means that the netowrk uses past experiences in the estimate of the Q-value target. This means that it keeps in mind what happend \textit{n} episodes ago and what the outcome was like. 

Sadly, even with experience replay, there are some environments where DQN diverges. 


This project aims to investigate where DQN diverges, why this happens, and how certain tricks such as experience replay and target network help to prevent divergence. 

# DQN: Where it fails and how to prevent this

https://pdfs.semanticscholar.org/7713/18e96c7df721135d92a78740f5ebf8696909.pdf


In reinforcement learning the agent needs to learn which actions to select in unknown, dynamic environments such that it will achieve its goal or maximise its objective function \cite{sutton2018reinforcement}. There are a few methods which help us achieve this goal, either on policy or off-policy. First, there are tabular methods. In this method we store a Q-value for each state action pair in a table and when we want to determine the Q-value of a given action in a certain state we simply look it up. However, in many tasks the state space is either continuous (resulting in an infinite number of states) or there are simply too many states (in which case the table becomes too large to store in memory). The other problem that can arise with a giant state space is that learning an Q-value for each state becomes impractical in the sense that we can't gather enough data for each state action pair to effectively learn the correct Q-value. To solve this problem we can instead use a function to approximate the correct Q-value, firstly the function allows us to use vastly less memory to store the required parameters and secondly the function could generalise allowing us to infer Q-value for state action pair even when we have no data for them.
    
In this blog we will focus on a specific case of function approximation for Q-learning called Deep Q-learning which uses a deep neural network to approximate the true Q-function. This method is called a Deep Q-network (DQN) and is based on Q-learning but replaces the tabular method from it with a deep neural network. In Q-learning, a state-action function is learned ($Q(s,a)$). Which is learned to approximate the potential future reward given a current state and an action we might perform. To derive a policy from this q-function we simply always take the action with the highest Q-value. In standard tabular Q-learning, a memory like table s created where all $Q(s,a)$ values are stored. With complex (video) games, simply using Q-learning is not an option. 

In comes Deep Learning!

If you combine the Q-learning from reinforcement learning and the deep neural networks as universal function approximators from deep learning, you will end up with a Deep Q-Network (DQN).
With DQN, the problem of the too large memory disappears, because the Q-value function is approximated (i.e. a function that determines the $Q(s,a)$ for each state is approximated).
As mentioned before, this is exactly what we need!


## On or Off Policy Learning

There are a few different methods that allow us to learn the Q-values for a given state we can either use an on-policy such as Monte Carlo learning, or an off policy method such as TD-learning, SARSA, or Q-learning.
The advantage of Monte Carlo learning is that the we learn the Q-value based on the actual policy which dictates the actions we take in each given state.
This often leads to much better convergence and is generally more stable than the off policy methods.
The advantage of the off-policy methods on the other hand is that we do not require the ability to actually finish an episode to learn something from the environment as the off-policy methods allow for online learning.
The trade-off here is that we require bootstrapping to estimate future rewards in lieu of having the actual rewards as in Monte Carlo training.
Luckily it has been proven that off policy methods will still converge to the true Q-value under the target policy \cite{sutton2018reinforcement}.
The problem however is that the deep neural networks that we want to use to approximate the Q-function is not guaranteed to converge to the global optima and therefore the previous convergence proofs fall by the wayside.

### The Deadly Triad
The risk of divergence is highest if all of the following three elements are present: 

* Function approximation
* Bootstrapping
* Off-policy training
        
These three elements are often called "the deathly triad" \cite{sutton2018reinforcement}.
Each one of these can have disastrous effects on whether or not the learned Q-function will approximate the true value under a given policy, and therefore often result in divergence of the trained Q-function ensuring that the greedy policy derived from it will be sub-optimal. 
        
The goal is to find various environments in which DQN and its various extensions show different convergence behaviours in the hope that analysing the differences in these environments can shine some light onto the problems that the various extension solve.
And if those correspond to the claims made by the researchers that introduced them.

## Investigated Extensions to Deep Q-Network

In this blog we'll look at two prominent extension to general framework of Deep Q-Networks, the two methods we will be investigating are Experience Replay \cite{lin1992self} and double Q-learning \cite{hasselt2010double}.
        
### Experience Replay
The goal of experience replay is to use the obtained training examples from the environment more efficiently.
In the basic Q-learning algorithm an experience obtained from the environment is only used once and then thrown away so we can move onto the next experience.
This is wasteful as some experience may be quite rare and others may be costly to obtain.
Therefore experiences should be used in an effective manner allowing for the algorithm to learn better.

The simplest method to achieve this is by simply storing passed experiences in memory and instead of learning from the current we simply learn from a sample of passed experiences instead.
This what the general form of Experience Replay entails.
And allows the agent to learn from experiences that it had before as if they had occur ed again.
However, \cite{lin1992self} notes that it is important for the environment to not change over time as that would invalidate previous experiences, resulting in past experiences becoming irrelevant or even harmful.
Furthermore it is noted that Experience Replay could be even more effect if the experience are replayed in reverse temporal order and the effectiveness can be further increased if a TD($\lambda$) method is used with a $\lambda$ greater than zero.
Here we will only look at the standard version that simply samples from past experiences.

\todo{Discuss if we want to include these, I did do some experiments with regards to Hindsight Experience Replay.}
There have been other improvements to Experience Replay that claim to boost the effectiveness of the extension such as Prioritized Experience Replay \cite{schaul2015prioritized}.
Here recently obtained experiences are given a higher priority when sampled so that the agent may be more likely to learn from these.
There is also Combined Experience Replay introduced by \cite{Zhang2017ADL}.
This method samples experiences from the past and then adds to this the latest obtained experience.
The hope here is to combat the influence of the size replay buffer (the list of past experiences that are stored) as they show that the size of this buffer influences the ability of the agent to effectively learn.

Another variant is Hindsight Experience Replay (HER) introduced by \cite{andrychowicz2017hindsight}.
It is introduced to solve a problem with games in which the goals keep changing and should allow the agent to learn from experiences obtained in the past in a manner similar to dynamic programming.
Consider the following scenario in which an agent starts in a random position in the world and has to move to another random "goal" position in the world, where the agent only receives a positive reward if it reaches the goal.
In this case we condition the action chosen in a state also on the goal that should be reached as in $\pi(a|s, g)$.
Instead of only storing the obtained experiences HER also stores an altered version of the experience.
After the agent has completed the episode and has not reached the goal, we consider the final state the agent reached as a pseudo goal and store experiences as if that were the intended goal all along.
This allows the agent to learn from episodes even if it were not able to reach the goal, and learn from sub-goals in a Dynamic Programming like fashion.

### Double Q-Learning
A different extension to Q-learning is the so called double Q-learning and is introduced in \cite{hasselt2010double}.
In Double Q-learning we augment the general Q-learning update step to include a critic like function that estimates Q-value for a given action in a certain state instead of also using the function that already chooses the action.
The reason for this is that Q-learning is an inherently biased estimator of the true Q-value under a given policy.
Therefore if we are particularly unlucky we only obtain experiences with favourable rewards while the real rewards we could obtain might be much worse.
Using two different Q-functions which learn from different experiences should prevent this from happening.

Another thing that using two different Q-functions solves is the problem of learning from an unstable function.
If during the learning phase the Q-values drastically change from update to update using a single Q-function to both update and estimate the current Q-value could lead to instability.

## Experiment Setup

### Environments

\todo{Add the other environments we used with reasoning.}
            
#### Bit-Twiddling

To show the effects of Hindsight Experience Replay we used a simple bit twiddling environment.
In this environment an agent is given a random string of bits as a starting position and another random string of bits representing the goal we want to achieve.
The agent can move from the starting position to the goal by toggling bits, and receives a reward of -1 for every action it takes.
When the agent reaches the goal state it receives a finishing reward of +1.
We chose this environment as we can use it as a simplified example of an environment in which a robot must move an object from one position in a room to another position in the room.
The problem in this environment is that the agent will often fail to reach the goal and therefore will have almost no rewards to actually learn from as it will only have negative rewards without a single positive reward.
We can solve this problem by adjusting how we view and or interact with the environment however these changes all involve assumptions which may not always hold.

First is the option of changing the state to include the goal.
Seeing as we are just trying to reach the goal state from the current state we might change the state instead to just be the difference between the current state and the goal state.
If we do this the game becomes much simpler as we always try to reach the same goal only from a different start state.
The problem with this is that we can no longer differentiate where in the state space we actually are only where we are relative to the goal state, this means that if some of the states are traps from which we incur a massive penalty we can no longer avoid them.
Therefore this is not a general solution.

Another option is change the reward to not be sparse and binary.
  If we change the reward to be the negative of the distance between the current state and the goal state the agent will be guided to reaching the goal state more effectively.

We no longer have the problem of not being able to distinguish between different states as with the previous option, however this can incur problems when the environment has a sort of blockade resulting in a shorter distance to the goal state might not mean we are heading in the right direction.
\todo{Add an example image of an environment in which this occurs to illustrate the problem.}
    
### Hyperparameters
#### Bit-Twiddling
#### Measurements
#### Bit-Twiddling

For the Bit-Twiddling environment we simply measure the difference between the minimal amount of steps that the agent should have to take to reach the goal state from the start state and the actual amount of steps that the agent took. This will tell us whether or not the agent learned to take the optimal path to the goal state and or learned anything at all. If one method produces a difference which is less than that the difference obtained from another method, the first method will be considered a better option.
        
## Results
## Analysis

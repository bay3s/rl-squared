###  RL^2 : Fast Reinforcement Learning via Slow Reinforcement Learning

**References**
- RL-squared (Paper) https://arxiv.org/abs/1611.02779


**Abstract**
- Deep RL has been successful, but the learning process requires a huge number of trials.
- RL-squared seeks to bridge this gap - rather than designing a fast RL algorithm proposes to represent it in a recurrent neural network and learn it from data.
- RNN receives all information a typical RL algorithm would receive including the state, reward, actions, and termination flags, and retains it state across episodes in a given MDP.
- The activations of the RNN store the state of the "fast" RL algorithm on the current previously unseen MDP.
- After training RL-squared shows performance on new MDPs close to human-designed algorithms with optimality guarantees.
- RL-squared on vision-based navigation task shows the ability to scale.


**Introduction**

*Bayesian RL*
- RL success seems to come at the expense of high sample complexity, this is largely due to the lack of a good prior which results in deep RL agents needing to rebuild knowledge about the world from scratch.
- Bayesian RL provides a framework for incorporating priors into the learning process, but exact computation of the Bayesian update is intractable in all but the simplest cases.
	- Practical Bayesian RL methods incorporate a mix of Bayesian and domain-specific ideas to bring down sample complexity.
	- But,they tend to make assumptions regarding the environment or become computationally intractable in high-dimensional settings.

*RL-Squared*
- Rather than hand-designing domain-specific RL algorithms, RL-squared views thee learning process of the agent itself as an objective which can be optimized using standard RL algorithms.
- The objective is averaged across possible MDPs according to a speicifc distribution whcih reflects the prior that we would like to distill into the agent.
- The agent is structured as an RNN which receives past rewards, actions, and termination flags as inputs in addition to the normally received observations.
- Internal state of the RNN is preserved across episodes, it has the capacity to perform learning in its own hidden activations.
- Thus, the learned algorithm also acts as the learning algorithm and can adapt to the task at hand when deployed.

*Fast RL via Slow RL*
- The paper suggests a different approach for designing better RL algorithms: instead of acting as the designers ourselves, learn the algorithm end-to-end using standard RL techniques.
- The fast RL algorithm is a computation whose state is stored in the RNN activations, and the RNN's weights are learned by a general-purpose slow RL algorithm.

**Formulation**
- RL-squared assumes knowledge of a set of MDPs and a distribution over them from which the MDPs can be sampled.
- A trial is a series of episodes of interaction with a fixed MDP, each trial may consist of multiple episodes.
- For each trial, a separate MDP is drawn from the distribution and for each episode within the trial a fresh starting state is drawn from the initial state distribution specific to the MDP.
- When an action is produce by the agent, the environment computes a reward, steps forward, computes the next state, and if the episode has terminated it sets the termination flag to 1.
- The next state, reward, and termination flag are concatenated to form the input to the recurrent policy which conditioned on the hidden state generates the next hidden state and action.
- The objective under this formulation is to maximize the total discounted reward accumulated during a single trial rather than a single episode (equivalently minimize the cumulative pseudo-regret).
- The underlying MDP changes across trials so as long as different strategies are required for different MDPS the agent must act differently accroding to its belief over which MDP it is currently in.
- *Note that this applies to MDPs as well as POMDPs.*


**Setup**

*Policy Representation*
- To alleviate the vanishing and exploding gradient problem GRUs are used which have good empirical performance.
- The output of the GRU is fed to a fully connected layer followed by a softmax function which forms the distribution over actions.

*Policy Optimization*
- Original paper uses the first-order implementation of TRPO because of its excellent empircal performance. and because it does not require excessive hyperparameter tuning.
- To reduce variance in the stochastic gradient estimation, a baseline which is also represented as an RNN using GRUs as building blocks is used.
- Optionally Generalized Advantage Esimtation (GAE) is used to reduce the variance further.

**Questions Addressed**
- Can RL-squared learn algorithms that achieve good performance on MDP classes with special structure, relative to existing algorithms tailored to this structure that have been proposed in the literature?
- Can RL-squared scale to high-dimensional tasks?

**Experiments**
- Experiments in the paper include comparative studies against algorithms trained for specific MDP tasks,
	- Multi-armed bandit strategies including,
		- Random
		- Gittins Index
		- UCB1
		- Thompson Sampling
		- $\epsilon$-Greedy
		- Greedy ($\epsilon$ = 0)
	- Tabular MDPs strategies including,
		- Random policy
		- PSRL which is a generalization of Thompson sampling to MDPs
		- BEB which is a model-based optimistic algorithm that includes an exploration bonus
		- UCRL2 which computes at each iteration the optimal policy against an optimistic MDP under the current belief using an extended value iteration
		- $\epsilon$-Greedy
		- Greedy ($\epsilon$ = 0)
	- Visual Navigation

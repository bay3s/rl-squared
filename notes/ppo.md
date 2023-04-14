### Proximal Policy Optimization Algorithms


### Intro

*Abstract*
- PPO alternates between sampling data through interaction with the environment, and optimizing a surrogate objective function using SGD.
- While standard policy gradient methods perform one gradient update per data sample, PPO methods enable multiple epochs of minibatch updates.
- PPO has the benefits of TRPO as far as data efficiency and reliable performance - but much simpler to implement, more general, and empirically better sample complexity.
- PPO outperforms other online policy gradient methods, strikes a favorable balance between sample complexity, simplicity, and wall time.


### Background

*Policy Gradient Methods (REINFORCE / VPG)*
- Policy gradient methods work by computing an estimator of the policy gradient and plugging it into a stochastic gradient ascent algorithm.

$$L^{P G}(\theta)=\hat{\mathbb{E}}_t\left[\log \pi_\theta\left(a_t \mid s_t\right) \hat{A}_t\right]$$

- The gradient estimator obtained by differentiating the objective function is given by,

$$\hat{g}=\hat{\mathbb{E}}_t\left[\nabla_\theta \log \pi_\theta\left(a_t \mid s_t\right) \hat{A}_t\right]$$

- It is appealing to perform multiple step sof otpimization on the loss $L^{PG}$ using the same trajectory, 
	- doing so is not well-justified
	- it empirically leads to large policy updates which can be destructive


*Trust Region Methods (TRPO)*
- The theory behind TRPO suggests that we maximize an objective function subject to a penalty term which takes into account the KL-divergence between the updated policy and the old policy,

$$
\underset{\theta}{\operatorname{maximize}} \hat{\mathbb{E}}_t\left[\frac{\pi_\theta\left(a_t \mid s_t\right)}{\pi_{\theta_{\text {old }}}\left(a_t \mid s_t\right)} \hat{A}_t-\beta \mathrm{KL}\left[\pi_{\theta_{\text {old }}}\left(\cdot \mid s_t\right), \pi_\theta\left(\cdot \mid s_t\right)\right]\right]
$$

- But, it is hard to choose a single value of $\beta$ that performs well across different problems or even a single problem where the cahracteristcs change over the course of learning.
- So, the penalty term is dropped in favor of a constraint that limits the on the size of the policy update.
- In practice, TRPO maximizes a "surrogate" objective function $L^{CPI}$ subject to a constraint on the size of the policy update (where $\hat{A}_t$ is the advantage estimate at time t),

$$
\begin{aligned}
\underset{\theta}{\operatorname{maximize}} \space & \hat{\mathbb{E}}_t\left[\frac{\pi_\theta\left(a_t \mid s_t\right)}{\pi_{\theta_{\text {old }}}\left(a_t \mid s_t\right)} \hat{A}_t\right] \\\\
\text { subject to } \space & \hat{\mathbb{E}}_t\left[\mathrm{KL}\left[\pi_{\theta_{\text {old }}}\left(\cdot \mid s_t\right), \pi_\theta\left(\cdot \mid s_t\right)\right]\right] \leq \delta .
\end{aligned}
$$

- This problem can be solved by using the conjugate gradient algorithm after making a linear approximation to the objective and a quadratic approximation to the constraint.


### PPO

*Clipped Surrogate Objective*
- Maximizing the surrogate objective $L^{CPI}$ as in the case of TRPO but without a constraint would lead to excessively large policy update.
- Suppose that the probability ration $r_t(\theta)$ is given by
$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}} (a_t | s_t)}$$
- Changes to the policy would move $r_t(\theta)$ away from 1 - PPO considers how to modify the objective to penalize these changes appropriately.
- PPO uses the objective (where $\epsilon$ is a hyperparameter),

$$
L^{C L I P}(\theta)=\hat{\mathbb{E}}_t\left[\min \left(r_t(\theta) \hat{A}_t, \space \operatorname{clip}\left(r_t(\theta), \space 1-\epsilon, \space 1+\epsilon\right) \hat{A}_t\right)\right]
$$

- The motivation behind this is to take the minimum of the clipped and the unclipped objective so that the final objective is a lowerbound (ie. a pessimistic bound) on the unclipped objective.
- The change in the probabilty ratio $r_t(\theta)$ is ignored if the objective improves by doing so but it is taken into account if a change in it makes the objective worse.


*Adaptive KL Penalty Coefficient*
- This approach can be used as an alternative to the clipped surrogate objective or as an addition to it.
- It imposes a penalty $\beta$ on the KL divergence, and adapts $\beta$ so that we achieve some target value of the KL divergence $d_{targ}$ at each policy update.
- *Step 1* Using several epochs of minibatch SGD, optimize the KL-penalized objective,

$$L^{K L P E N}(\theta)=\hat{\mathbb{E}}_t\left[\frac{\pi_\theta\left(a_t \mid s_t\right)}{\pi_{\theta_{\text {old }}}\left(a_t \mid s_t\right)} \hat{A}_t-\beta \mathrm{KL}\left[\pi_{\theta_{\text {old }}}\left(\cdot \mid s_t\right), \pi_\theta\left(\cdot \mid s_t\right)\right]\right]$$
 
- *Step 2* Compute $d = \hat{\mathbb{E}}[KL[\pi_{\theta_{old}}(.|s_t), \space \pi_\theta(.|s_t)]]$
	- if $d < d_{targ} / 1.5, \beta \leftarrow \beta / 2$
	- if $d > d_{targ} \times 1.5, \beta \leftarrow \beta / 2$


*Practical Implementation Details*
- For implementations using a package supporting autodiff, you can simply consturct the loss $L^{CLIP}$ or $L^{KLPEN}$ in place of the policy gradient objective.
- Techniques for computing the variance-reduced advantage function estimators can also be used in conjunction,
	- Generalized Advantage Estimators
	- Finite-Horizon Estimators
- In the case where there is parameter sharing between the policy and value function, the loss function should take into account the policy surrogate and the value function error,

$$L_t^{C L I P+V F+S}(\theta)=\hat{\mathbb{E}}_t\left[L_t^{C L I P}(\theta)-c_1 L_t^{V F}(\theta)+c_2 S\left[\pi_\theta\right]\left(s_t\right)\right]$$


#### References
- Proximal Policy Optimization Algorithms (Paper) https://arxiv.org/abs/1707.06347

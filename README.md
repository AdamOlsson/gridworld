# Reinforcement Learning Part 1

<p>Part 1 of my Reinforcement Learning (RL) series. During this series, I dwell into the field of RL by applying various methods to video games to learn and understand how an algorthm can learn to play by itself. The motivation for doing this series is simply by pure interest and to gain knowledge and experience in the field of Machine Learning.

The litterature follow throughout this series is <em>Reinforcement Learning</em> "An Introduction" by Ricard S. Button and Andrew G. Barto. 
ISBN: 9780262039246
</p>

## Dynamic Programming
<p>Dynamic programming (DP) refers to a collection of algorithm that given a perfect model of the environment as a Markov Decision Process can compute the optimal policy. In other words, if have full knowledge of hour system, or in this case game, we can compute the best way to play the game. However, this comes with great computational expense because it basically requires the algorithm to iterate over all states of the environment which quickly can become an unfeasable task.
</p>

### Policy Evaluation
<p>To evaluate how well a policy would perform in the environment we compute the state-value function <em>v<sub>π</sub></em> under the policy π. For each state we iterate through all states, we look at the probability of taking action <em>a</em> in state <em>s</em> under the policy π, the transition function <em>p</em> and the discounted future rewards to calculate the value in the state, <em>V(s)</em>. We continute to do this until the improvement between iterations is below a threshold.
</p>

<p align="center">
<em>V(s) = &sum;<sub>a</sub>π(a|s)&sum;<sub>s', r </sub>p(s', r| s, a)[r + &gamma; V(s')]</em>
</p>

<p>Below you can see the result of a value iteration on a Gridworld with terminal states top left and bottom right. The basic idea is for an agent playing the game to get to the terminal state as fast as possible. Each action, or step, in the world rewards the agent with -1 point. In the image we see that the value of the bottom left and top right corners are extremely low and the closer we get to the terminal state, the better value the state has.</p>

<p align="center">
<img align="center" src=https://github.com/AdamOlsson/rl_policy_iteration/blob/master/img/Heatmap_default.png>
</p>

### Policy Improvement
<p>Once the value function has been determined we can start improving our policy. Suppose that we have calculated our value function under a policy π and for some state <em>s</em> we would like to know if we should take a different action to yield a better value function. Since we have complete knowledge of our system, due to this being a DP problem, we can simply try to take another action in the state. Then we simply greedily select the action that gave the highest score and update our policy.</p>

<p align="center"><em>π(s) = </em> argmax<sub><em>a</em></sub> <em>&sum;<sub>s', r </sub>p(s', r| s, a)[r + &gamma; V(s')]</em></p>

### Policy Iteration
<p>Now we have a method of evaluate our policy as well as updating our existing policy to make it better. Merging this two methods is called Policy Iteration. The idea is that we first evaluate our policy and then try to updating it in an iterative process to yield an optimal policy. The image below shows the result of the same Gridworld as in Policy Evaluation.</p>

<p align="center">
<img align="center" src=https://github.com/AdamOlsson/rl_policy_iteration/blob/master/img/Heatmap_policy_iteration.png>
</p>

### Value Iteration
<p>A drawback of Policy Iteration is that the current policy needs to be evaluated during every iteration. The evaluation of a policy can be a computational heavy task. In Value Iteration we shorten the computation of the Policy Evaluation by only allowing one single iteration through all states. The reasoning behind this is that during Policy Evaluation, that we do not need to wait for it completely because after the first few iteration very little change to the value function is seen. The relation between the states is defined quite early, i.e its quickly realised that some states are better than others, just not how much better. Therefor we can change the update of our state-value:</p> 

<p align="center">
  <em>V(s) = </em> max<sub><em>a</em></sub> <em>&sum;<sub>s', r </sub>p(s', r| s, a)[r + &gamma; V(s')]</em>
</p>

The resulting state-value function is the same as in Policy Iteration. Below you see the final policy using value iteration.

<p align="center">
<img align="center" src=https://github.com/AdamOlsson/rl_policy_iteration/blob/master/img/actions_value_iteration.png>
</p>

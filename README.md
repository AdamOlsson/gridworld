# Reinforcement Learning Part 1

<p>Part 1 of my Reinforcement Learning (RL) series. During this series, I dwell into the field of RL by applying various methods to video games to learn and understand how an algorthm can learn to play by itself. The motivation for doing this series is simply by pure interest and to gain knowledge and experience in the field of Machine Learning.

The litterature follow throughout this series is <em>Reinforcement Learning</em> "An Introduction" by Ricard S. Button and Andrew G. Barto. 
ISBN: 9780262039246
</p>

## Dynamic Programming
Dynamic programming refers to a collection of algorithm that given a perfect model of the environment as a Markov Decision Process can compute the optimal policy. In other words, if have full knowledge of hour system, or in this case game, we can compute the best way to play the game. However, this comes with great computational expense because it basically requires the algorithm to iterate over all states of the environment which quickly can become an unfeasable task.

### Policy Evaluation
To evaluate how well a policy would perform in the environment we compute the state-value function <em>v<sub>$\pi$</sub></em> under the policy $\pi$.
import numpy as np 
from gridworld_env import GridWorldEnv
from policy_eval import plot_vfn

def value_iteration(env, discount_factor=1.0, epsilon=0.0001):

    def one_step_lookahead(s, vfn):
        actions = np.zeros(env.nA)

        for a in range(env.nA):
            [prob, next_state, reward, done] = env.P[s][a]
            actions[a] = prob*(reward + discount_factor*vfn[next_state[0]*env.shape[0] + next_state[1]])

        return actions

    policy = np.zeros([env.nS, env.nA])
    vfn = np.zeros(env.nS)

    while True:
        delta = 0

        for s in range(env.nS):
            
            action_values = one_step_lookahead(s, vfn)

            best_action_value = max(action_values)

            delta = max(delta, abs(vfn[s] - best_action_value))

            vfn[s] = best_action_value

            best_action = np.argmax(action_values)

            policy[s] = np.eye(env.nA)[best_action]

        if delta < epsilon:
            return policy, vfn

def plot_policy(p, env):

    p_actions = []
    for a in np.argmax(p, axis=1):
        p_actions.append(env.actions[a])

    # Ugly but works
    print('\n   Actions taken following the final policy:\n   (top-left && bottom-right are terminal states)\n\n   {}   {}   {}   {}'.format(p_actions[0], p_actions[1],p_actions[2],p_actions[3]))
    print('   {}   {}     {}     {}'.format(p_actions[4], p_actions[5],p_actions[6],p_actions[7]))
    print('   {}   {}     {}  {}'.format(p_actions[8], p_actions[9],p_actions[10],p_actions[11]))
    print('   {}   {}  {}  {}\n'.format(p_actions[12], p_actions[13],p_actions[14],p_actions[15]))


if __name__ == "__main__":
    env = GridWorldEnv()

    p, v = value_iteration(env)

    plot_policy(p, env)
    plot_vfn(np.reshape(v, (4,4)))
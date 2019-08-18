from gridworld_env import GridWorldEnv
from policy_eval import policy_eval, plot_vfn
import numpy as np

def policy_improvement(policy, env, policy_eval_fn=policy_eval, discount_factor=1.0):

    def one_step_lookahead(s, vfn):
        actions = np.zeros(env.nA)

        for a in range(env.nA):
            [prob, next_state, reward, done] = env.P[s][a]
            actions[a] = prob*(reward + discount_factor*vfn[next_state[0]*env.shape[0] + next_state[1]])
        return actions

    action_values = np.zeros(env.nA)

    while True:
        vfn = policy_eval_fn(policy, env, discount_factor=1)
        policy_stable = True

        for state in range(env.nS):
            action_values = one_step_lookahead(state, vfn)

            best_action = np.argmax(action_values)

            current_action = np.argmax(policy[state])

            if current_action != best_action:
                policy_stable = False

            policy[state] = np.eye(env.nA)[best_action]

        if policy_stable:
            return vfn, policy

def plot_policy(p, env):

    p_actions = []
    for a in np.argmax(better_policy, axis=1):
        p_actions.append(env.actions[a])

    # Ugly but works
    print('\n   Actions taken following the final policy:\n   (top-left && bottom-right are terminal states)\n\n   {}   {}   {}   {}'.format(p_actions[0], p_actions[1],p_actions[2],p_actions[3]))
    print('   {}   {}     {}     {}'.format(p_actions[4], p_actions[5],p_actions[6],p_actions[7]))
    print('   {}   {}     {}  {}'.format(p_actions[8], p_actions[9],p_actions[10],p_actions[11]))
    print('   {}   {}  {}  {}\n'.format(p_actions[12], p_actions[13],p_actions[14],p_actions[15]))


if __name__ == '__main__':
    env = GridWorldEnv()
    random_policy = np.ones([env.nS, env.nA]) / env.nA
    v, better_policy = policy_improvement(random_policy, env)

    plot_policy(better_policy, env)
    plot_vfn(np.reshape(v, (4,4)), 'Value map after policy iteration')
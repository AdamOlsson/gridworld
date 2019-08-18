from gridworld_env import GridWorldEnv
from policy_eval import policy_eval
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


if __name__ == '__main__':
    env = GridWorldEnv()
    random_policy = np.ones([env.nS, env.nA]) / env.nA
    v, better_policy = policy_improvement(random_policy, env)

    print(better_policy)
    print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape( np.argmax(better_policy, axis=1), env.shape) )
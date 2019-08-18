import numpy as np

class GridWorldEnv():
    def __init__(self):
        self.actions = ('up', 'right', 'down', 'left')
        self.nA = len(self.actions)
        self.world = np.zeros((4,4))
        self.nS = np.size(self.world)

        self.shape = np.shape(self.world)

        state_win = [(0,0), (3,3)]
        state_loose = []

        self.P = self.genAllTransitions(self.world, state_win, state_loose)


    def genAllTransitions(self, world, terminal_state, loose_state):

        def transition(pos, action):
            if action == 'up':
                new_pos = (max(0, pos[0]-1), pos[1])
            elif action == 'right':
                new_pos = (pos[0], min(3, pos[1]+1))
            elif action == 'left':
                new_pos = (pos[0], max(0, pos[1]-1))
            else: # if action == 'down':
                new_pos = (min(3, pos[0]+1), pos[1])
        
            return new_pos

        transitions = []
        for r in range(np.shape(world)[0]):
            for c in range(np.shape(world)[1]):
                transition_state = []
                for a in self.actions:
                    current_state = (r, c)
                    next_state = transition(current_state, a)
                    
                    if current_state in terminal_state or current_state in loose_state:
                        transition_prob = 0
                        next_state = current_state
                    else:
                        transition_prob = 1
                        
                    done = current_state in terminal_state or current_state in loose_state
                    #reward = -1 if not current_state in terminal_state else 0

                    reward = -1
                    if current_state in terminal_state or current_state in loose_state:
                        reward = 0
                    elif next_state in loose_state:
                        reward = -10

                    transition_state.append((transition_prob, next_state, reward, done))
                    #transition_state.append((transition_prob, current_state, next_state, reward, done, a)) # debug

                transitions.append(transition_state)
        
        return transitions
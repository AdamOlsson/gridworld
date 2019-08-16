import numpy as np

class GridWorldEnv():
    def __init__(self):
        self.actions = ('up', 'right', 'left', 'down')
        self.nA = len(self.actions)-1
        self.world = np.zeros((4,4))
        self.nS = np.size(self.world)

        self.P = self.genAllTransitions(self.world, (3,3))

    def transition(self, pos, action):
        if action == 'up':
            new_pos = (max(0, pos[0]-1), pos[1])
        elif action == 'right':
            new_pos = (pos[0], min(3, pos[1]+1))
        elif action == 'left':
            new_pos = (pos[0], max(0, pos[1]-1))
        else: # if action == 'down':
            new_pos = (min(3, pos[0]+1), pos[1])
        
        return new_pos

    def genAllTransitions(self, world, terminal_state):
        transitions = []
        for r in range(np.shape(world)[0]):
            for c in range(np.shape(world)[1]):
                transition_state = []
                for a in self.actions:
                    current_state = (c,r)
                    next_state = self.transition(current_state, a)

                    transition_prob = 0 if current_state == next_state else 1
                    
                    if current_state == terminal_state:
                        transition_prob = 0
                        next_state = terminal_state

                    done = next_state == terminal_state
                    reward = -1 if not next_state == terminal_state else 0

                    transition_state.append((transition_prob, next_state, reward, done))
                transitions.append(transition_state)
        
        return transitions
                         

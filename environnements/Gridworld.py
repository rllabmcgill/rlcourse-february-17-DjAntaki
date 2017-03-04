import numpy as np
from environnements import Env
from itertools import product


def get_grid1():
    a = np.array([[0,0,0,0,0,0,0,0,0,0,0,0],
              [0,1,1,1,1,1,1,1,1,1,1,0],
              [0,0,0,0,0,0,0,0,0,0,0,0]])
    return a

def test1():
    env = Gridworld()
    env.take_action(0,verbose=True)
    env.take_action(0,verbose=True)
    for i in range(11):
        env.take_action(2,verbose=True)

    assert env.is_terminal_state()


class Gridworld(Env):
    def __init__(self, grid_func=get_grid1):
        self.action_space = {0:'u',1:'l',2:'r',3:'d'}
        self.num_actions = 4
        self.grid = grid_func()
        a,b  = self.grid.shape
        self.state_space = list(filter(lambda x:self.grid[x[0],x[1]] == 0 , product(range(a),range(b))))
        self.id_to_state = {i:j for i,j in enumerate(self.state_space)}
        self.state_to_id = {j:i for i,j in self.id_to_state.items()}
        self.end_states = list(map(lambda x:self.state_to_id[x],[(0, self.grid.shape[1]-1)]))
        self.num_states = len(self.state_space)
        self.restart()

    def restart(self):
        a,b  = self.grid.shape
        self.current_state = self.state_to_id[(a-1,0)]

    def take_action(self, action, update_state=True, verbose=False):
        assert action in (0,1,2,3)
        action = self.action_space[action]

        i,j = self.id_to_state[self.current_state]
        next_state = (i,j)
        if action == 'u':
            if i - 1 >= 0 and self.grid[i-1,j] == 0 :
                next_state = (i-1,j)
        elif action == 'l':
            if j - 1 >= 0 and self.grid[i,j-1] == 0 :
                next_state = (i,j-1)
        elif action == 'r':
            if j + 1 < self.grid.shape[1] and self.grid[i,j+1] == 0 :
                next_state = (i,j+1)
        elif action == 'd':
            if i + 1 < self.grid.shape[0] and self.grid[i+1,j] == 0 :
                next_state = (i+1,j)
        else :
            raise NotImplemented()

        reward = -1
        if self.is_terminal_state():
            reward = 1000

        next_state = self.state_to_id[next_state]
        if update_state:
            self.current_state = next_state

        if verbose:
            g = np.array(self.grid)
            print(action)
            p = self.id_to_state[next_state]
            g[p[0],p[1]] = 5
            print(g)

        return next_state, reward


if __name__ == "__main__":
    test1()
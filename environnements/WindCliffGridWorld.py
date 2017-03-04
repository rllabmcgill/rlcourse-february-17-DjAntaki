import numpy as np
from environnements import Env
from matplotlib import pyplot as plt
from itertools import product

def get_grid2():
    a = np.array([[0,0,0,1,0,0,0,1,0,0,0,0],
                  [0,1,0,1,0,1,0,1,0,1,1,0],
                  [0,1,0,1,0,1,0,1,0,1,1,0],
                  [0,1,0,0,0,1,0,0,0,0,0,0],
                  [0,1,1,1,0,1,1,1,0,1,1,0],
                  [0,3,0,0,0,3,0,0,0,3,0,0],
                  [0,2,2,2,2,2,2,2,2,2,2,0]])        
    return a

def get_grid4():
    a = np.array([[0,0,0,1,0,0,0,1,0,0,0,0],
                  [0,1,0,1,0,1,0,1,0,1,1,0],
                  [0,1,0,0,0,1,0,0,0,1,1,0],
                  [0,1,0,1,0,1,1,1,0,0,0,0],
                  [0,3,0,3,0,3,0,0,0,3,3,0],
                  [0,2,2,2,2,2,2,2,2,2,2,0]])
    return a

def get_grid5():
    a = np.array([[0,0,0,0,0,0,0,0,0,0,0,0],
                  [0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
                  [0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
                  [0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
                  [0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
                  [0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
                  [0,3,3,3,3,3,3,3,3,3,3,0],
                  [0,3,3,3,3,3,3,3,3,3,3,0],
                  [0,2,2,2,2,2,2,2,2,2,2,0]])
    return a

def get_grid3():
    a = np.array([[0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,3,3,3,3,3,3,3,3,3,3,0],
                  [0,3,3,3,3,3,3,3,3,3,3,0],
                  [0,2,2,2,2,2,2,2,2,2,2,0]])
    return a

def get_grid6():
    a = np.array([[1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
[1, 0, 0, 1, 2, 2, 2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
[1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0],
[0, 0, 0, 1, 1, 1, 0, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 1, 0, 1, 0, 2, 2, 2, 2, 2, 2, 1, 0, 1, 1, 2, 1, 0],
[0, 1, 0, 0, 0, 1, 0, 2, 2, 2, 2, 2, 2, 0, 0, 1, 2, 3, 1, 0],
[0, 1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0],
[0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 0],
[0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
[0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
[0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 2, 1, 0, 1, 1, 0, 1, 1, 0],
[0, 1, 0, 1, 0, 1, 0, 2, 1, 0, 1, 3, 1, 0, 0, 1, 0, 1, 0, 0],
[0, 1, 0, 1, 1, 1, 0, 3, 3, 0, 1, 3, 1, 0, 0, 1, 0, 1, 0, 1],
[0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 3, 1, 1, 0, 1, 0, 1, 0, 1],
[0, 1, 0, 0, 0, 1, 1, 1, 3, 3, 3, 3, 3, 3, 0, 0, 0, 1, 0, 1],
[0, 1, 1, 1, 0, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0],
[0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 0, 1, 1, 0],
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 1, 0]])
    return a

def get_grid7():
    a = np.array([[1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 3, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
[1, 3, 0, 1, 2, 2, 2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
[1, 1, 0, 1, 3, 3, 3, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0],
[0, 0, 0, 1, 1, 1, 0, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 1, 0, 1, 0, 2, 2, 2, 2, 2, 2, 1, 0, 1, 1, 2, 1, 0],
[0, 1, 0, 0, 0, 1, 0, 2, 2, 2, 2, 2, 2, 0, 0, 1, 2, 3, 1, 0],
[0, 1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0],
[0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 0],
[0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
[0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 3, 3, 3, 0, 1, 0, 0, 0, 0, 0],
[0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 2, 2, 3, 0, 1, 1, 0, 1, 1, 0],
[0, 1, 0, 1, 0, 1, 0, 2, 1, 0, 2, 3, 2, 0, 0, 1, 0, 1, 0, 0],
[0, 1, 0, 1, 1, 1, 0, 3, 3, 0, 1, 3, 1, 0, 0, 1, 0, 1, 0, 1],
[0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 2, 3, 2, 1, 0, 1, 0, 1, 0, 1],
[0, 1, 0, 0, 0, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 0, 0, 1, 0, 1],
[0, 1, 1, 1, 0, 3, 3, 3, 3, 2, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0],
[0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 0, 1, 1, 0],
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 1, 0]])
    return a
def get_grid8():
    a = np.array([[1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 3, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
[1, 3, 0, 1, 2, 2, 2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
[1, 1, 0, 1, 3, 3, 3, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
[0, 0, 0, 1, 1, 1, 0, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 1, 0, 1, 0, 2, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 0],
[0, 1, 0, 0, 0, 1, 3, 2, 1, 1, 1, 1, 1, 0, 0, 1, 2, 3, 1, 0],
[0, 1, 2, 2, 2, 1, 3, 3, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0],
[0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 0],
[0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
[0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 3, 3, 3, 0, 1, 0, 0, 0, 0, 0],
[0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 2, 2, 3, 0, 1, 1, 0, 1, 1, 0],
[0, 1, 0, 1, 0, 1, 0, 2, 1, 0, 2, 3, 2, 0, 0, 1, 0, 1, 0, 0],
[0, 1, 0, 1, 1, 1, 0, 3, 3, 0, 1, 3, 1, 0, 0, 1, 0, 1, 0, 1],
[0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 2, 3, 2, 1, 0, 1, 0, 1, 0, 1],
[0, 1, 0, 0, 0, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 0, 0, 1, 0, 1],
[0, 1, 1, 1, 0, 3, 3, 3, 3, 2, 1, 1, 1, 1, 2, 1, 0, 1, 0, 0],
[0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 0, 1, 1, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 2, 2, 0, 0, 1, 0]])
    return a

def get_grid9():
    a = np.array([[1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 3, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
[1, 3, 0, 1, 2, 2, 2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
[1, 1, 0, 1, 3, 3, 3, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
[0, 0, 0, 1, 1, 1, 0, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 1, 0, 1, 0, 2, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 0],
[0, 1, 0, 0, 0, 1, 3, 2, 1, 1, 1, 1, 1, 0, 0, 1, 2, 3, 1, 0],
[0, 1, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0],
[0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 0],
[0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
[0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 3, 3, 3, 0, 1, 0, 0, 0, 0, 0],
[0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 3, 2, 3, 0, 1, 1, 0, 1, 1, 0],
[0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 2, 3, 2, 0, 0, 1, 0, 1, 0, 0],
[0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 3, 1, 0, 0, 1, 0, 1, 0, 1],
[0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 2, 3, 2, 1, 0, 1, 0, 1, 0, 1],
[0, 0, 0, 0, 0, 1, 1, 2, 3, 3, 0, 3, 3, 3, 3, 0, 0, 1, 0, 1],
[0, 1, 1, 1, 0, 3, 3, 3, 0, 2, 1, 1, 1, 1, 2, 1, 0, 1, 0, 0],
[0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 0, 1, 1, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 2, 2, 0, 0, 1, 0]])
    return a

def plot_grid(g,title) :
    img2 = plt.imshow(g, interpolation='nearest',
                      origin='lower')
    plt.yticks(np.arange(0, g.shape[0], 1.0))
    plt.xticks(np.arange(0, g.shape[1], 1.0))

    plt.title(title)
    plt.show()

if __name__ == "__main__":
    plot_grid(np.flipud(get_grid2()),"map #2")
    plot_grid(np.flipud(get_grid3()),"map #1")
    plot_grid(np.flipud(get_grid5()),"map #3")
    plot_grid(np.flipud(get_grid9()), "map #4")
class WindCliffGridWorld(Env):
    """
    
    
    the start position is the bottom left corner
    the end position is the bottom right corner
    
    interpretation of grid value (harcoded)
    - 0 : normal tile
    - 1 : wall
    - 2 : cliff
    - 3 : windy tile type 1
    
    """
    valid_state_tiles = [0,3]
    valid_move_tiles = [0,2,3]
    special_type_tiles = [2,3]
    
    def __init__(self, grid_func=get_grid2,map_name=""):
        self.map_name = map_name
        self.action_space = {0:'u',1:'l',2:'r',3:'d'}
        self.num_actions = 4
        self.grid = grid_func()
        a,b  = self.grid.shape
        self.state_space = list(filter(lambda x:self.grid[x[0],x[1]] in self.valid_state_tiles , product(range(a),range(b))))
        self.id_to_state = {i:j for i,j in enumerate(self.state_space)}
        self.state_to_id = {j:i for i,j in self.id_to_state.items()}

        self.end_states = [self.state_to_id[(self.grid.shape[0] - 1, self.grid.shape[1] - 1)]]
        #self.end_states = list(map(lambda x:self.state_to_id[x],[(self.grid.shape[0]-1, self.grid.shape[1]-1)]))
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
            if i - 1 >= 0 and self.grid[i-1,j] in self.valid_move_tiles :
                next_state = (i-1,j)
        elif action == 'l':
            if j - 1 >= 0 and self.grid[i,j-1] in self.valid_move_tiles :
                next_state = (i,j-1)                        
        elif action == 'r':
            if j + 1 < self.grid.shape[1] and self.grid[i,j+1] in self.valid_move_tiles :
                next_state = (i,j+1)            
        elif action == 'd':
            if i + 1 < self.grid.shape[0] and self.grid[i+1,j] in self.valid_move_tiles :
                next_state = (i+1,j)
        else :
            raise NotImplemented()
            
        reward = -0.5
        tile_type = self.grid[next_state]

        if tile_type in self.special_type_tiles:
            while True:
                if tile_type == 3:
                    #Wind application
                    if np.random.binomial(1,0.5):
                        direction = np.random.choice([0,1,2,3])
                        delta_move = {0:(-1,0),1:(1,0),2:(0,-1),3:(0,1)}[direction]
                        potential_next_state = (next_state[0]+delta_move[0],next_state[1]+delta_move[1])
                        if self.grid[potential_next_state] in self.valid_move_tiles:
                            next_state = potential_next_state
                            tile_type = self.grid[next_state]
                    else :
                        break
                elif tile_type == 2:
                    # Falling in cliff
                    reward = -2
                    next_state = (self.grid.shape[0] - 1, 0)  # going back to starting position
                    break
                elif tile_type in (0,):
                    break
                else :
                    raise NotImplementedError("Tile type %i not recognized"%tile_type)

                if tile_type == 2:
                    #Falling in cliff
                    reward = -2
                    next_state = (self.grid.shape[0]-1,0) #going back to starting position

        next_state = self.state_to_id[next_state]
        if update_state:
            self.current_state = next_state

        if self.is_terminal_state():
            reward = 100

        if verbose:
            g = np.array(self.grid)
            print(action)
            p = self.id_to_state[next_state]
            g[p[0],p[1]] = 5
            print(g)
        
        return next_state, reward

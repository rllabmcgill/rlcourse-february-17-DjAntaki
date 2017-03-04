from environnements.WindCliffGridWorld import *
from algorithms import Q_learning
from utils import kfold_decorator

#env = WindCliffGridWorld()
#env = WindCliffGridWorld(get_grid5)
env = WindCliffGridWorld(get_grid9)

nfolds = 3
Q_learning = kfold_decorator(Q_learning,nfolds,show_graph=True)

nb_episode=10000
max_iter=1500
print("Q learning")
Q_learning(env,nb_episode=nb_episode,max_iter=max_iter,label="Q learning")
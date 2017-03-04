from environnements.WindCliffGridWorld import *
from algorithms import Q_learning,Q_learning2
from algorithms.treebackup import Treebackup
from utils import kfold_decorator

nfolds = 3
Q_learning = kfold_decorator(Q_learning,nfolds,show_graph=True)
Q_learning2 = kfold_decorator(Q_learning2,nfolds,show_graph=True)
Treebackup = kfold_decorator(Treebackup,nfolds,show_graph=True)
nb_episode=1000
max_iter = 750

out = ""

for map_name, grid_func in [("map#1",get_grid3),("map#2",get_grid2),("map#3",get_grid5),("map#4",get_grid9)]:
    env = WindCliffGridWorld(grid_func,map_name)
    if map_name == "map#4":
        nb_episodes, max_iter = 2000,1500
    for label, algorithm in [("Q learning",Q_learning),("Expected Sarsa",Q_learning2),("2-steps treebackup",Treebackup)]:
        print(label)
        qvalues, reward, it_taken = algorithm(env,nb_episode=nb_episode,max_iter=max_iter,label=label)
        print(np.sum(it_taken),np.average(reward[-100:]))
        out += " & ".join([label, str(round(np.sum(it_taken),2)),str(round(np.average(reward[-100:]),2))]) + "\\ \n"
    out += "\n"

print(out)
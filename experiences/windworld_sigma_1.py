from environnements.WindCliffGridWorld import *
from algorithms.qsigma import *
from utils import kfold_decorator
from plots.plot import make_plots

nb_episode=1000
max_iter = 750
test=False
Q_sigma = kfold_decorator(Q_sigma,3,show_graph=True)
line_names = ["fix0.1","fix0.9","fix0.5","uniform","binomial0.5","truncatedGauss","TypeBased"]
for map_name, grid_func in [("map#1",get_grid3),("map#2",get_grid2),("map#3",get_grid5)]:
#for map_name, grid_func in [("map#1",get_grid3),("map#2",get_grid2),("map#3",get_grid5),("map#4",get_grid9)]:
    env = WindCliffGridWorld(grid_func,map_name)
    if map_name == "map#4":
        nb_episodes, max_iter = 2000,1500
    for n in [2]:
        result = []
        if test :
            Q_sigma(env, n, nb_episode = nb_episode,label="Q(sigma)")
        #        Q_sigma(env,n,nb_episode=nb_episode)
        else :
            print("%i fixed_sigma_01"%n)
            rez = Q_sigma(env, n, get_sigma_func=fixed_sigma_func(0.1),nb_episode=nb_episode,max_iter=max_iter,label="Q(sigma) - "+line_names[0])
            result.append(rez)
            print("%i fixed_sigma_09"%n)
            rez = Q_sigma(env, n, get_sigma_func=fixed_sigma_func(0.9),nb_episode=nb_episode,max_iter=max_iter,label="Q(sigma) - "+line_names[1])
            result.append(rez)
            print("%i fixed_sigma_05"%n)
            rez = Q_sigma(env, n, get_sigma_func=fixed_sigma_func(0.5),nb_episode=nb_episode,max_iter=max_iter,label="Q(sigma) - "+line_names[2])
            result.append(rez)
            print("%i uniform_sigma_1" % n)
            rez = Q_sigma(env, n, get_sigma_func=uniform_sigma_func(), nb_episode=nb_episode,max_iter=max_iter,label="Q(sigma) - "+line_names[3])
            result.append(rez)
            print("%i binomial_sigma_1" % n)
            rez = Q_sigma(env, n, get_sigma_func=binomial_sigma_func(0.5), nb_episode=nb_episode,max_iter=max_iter,label="Q(sigma) - "+line_names[4])
            result.append(rez)
            print("%i truncated_sigma_1" % n)
            rez = Q_sigma(env, n, get_sigma_func=truncated_gaussian_func(), max_iter=max_iter, nb_episode=nb_episode,
                                                 label="Q(sigma) - "+line_names[5])
            result.append(rez)
            print("%i harcoded_sigma_1" % n)
            rez = Q_sigma(env, n, get_sigma_func=hardcoded_sigma_windyworld, nb_episode=nb_episode, max_iter=max_iter,
                                                 label="Q(sigma) - "+line_names[6])
            result.append(rez)
            rewards = list(map(itemgetter(1),result))
            it_taken = list(map(itemgetter(2), result))
            make_plots(it_taken,rewards,line_names,10,map_name=map_name,saveplot=False)

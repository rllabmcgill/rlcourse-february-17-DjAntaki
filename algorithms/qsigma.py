from utils import *
from operator import itemgetter


def fixed_sigma_func(v):
    def get_sigma(*args, **kwargs):
        return v
    return get_sigma

def binomial_sigma_func(p):
    from numpy.random import binomial
    def binomial_sample(*args, **kwargs):
        return binomial(n=1,p=p)
    return binomial_sample

def uniform_sigma_func():
    from numpy.random import uniform
    def uniform_sample(*args, **kwargs):
        return uniform(0,1,None)
    return uniform_sample

def truncated_gaussian_func(mean=0.5,var=0.8,lower_limit=0,upper_limit=1):
    from numpy.random import normal
    def gaussian_sample(*args, **kwargs):
        sample = normal(loc=mean, scale=var)
        if sample < lower_limit:
            return lower_limit
        elif sample > upper_limit:
            return upper_limit
        return sample
    return gaussian_sample

def hardcoded_sigma_windyworld(env,*args,**kwargs):
    sigma_map = {0:0.9,3:0.1}
    return sigma_map[env.grid[env.id_to_state[env.current_state]]]

def hardcoded_sigma_windyworld2(env,*args,**kwargs):
    if any(env.get_neighbours(env.current_state)):
        return NotImplementedError() #TODO
    sigma_map = {0:0.9,3:0.1}
    return sigma_map[env.grid[env.id_to_state[env.current_state]]]


def Q_sigma(env, n, fixed_mu=False, fixed_pi=False, alpha=0.05, gamma=0.9, epsilon=5e-2, softmax_temp=10, nb_episode=1000, get_sigma_func=fixed_sigma_func(0), max_iter=1500,label="Q(sigma)",verbose=False,show_graph=False,evaluate_pi=False):
    """
    Off-policy n-step Q(sigma) for estimating Q ~ q*, or Q ~ q_pi for a given pi
    """
    assert (gamma > 0 and gamma < 1)
    assert (type(n) is int and n > 0)
#    if not scheduled_epsilon_decrease is False :
#        epsilon_step = 0

    qvalues = np.ones((env.num_states, env.num_actions))
    qvalues[env.end_states[0]] = np.zeros((4,)) #Little fix because the end state is never updated
    stats = []

    if not(fixed_pi is False):
        policy = fixed_pi
    else :
        policy = np.array([normalize(x,softmax_temp) for x in qvalues])

    for i in range(nb_episode):
        T = float('inf')
        t = -1
        total_reward = 0
        env.restart()

#        if not scheduled_epsilon_decrease is False :
#            if i > scheduled_epsilon_decrease[epsilon_step][0]:
#                epsilon_step += 1
#                epsilon *= 0.5

        Q = np.zeros((n,))
        states = np.zeros((n,),dtype=np.int)
        deltas = np.zeros((n,))
        actions = np.zeros((n,),dtype=np.int)
        sigmas = np.zeros((n,))
        ros = np.zeros((n,))
        pis = np.zeros((n,))

        if fixed_mu is False:
            mu = normalize(qvalues[env.current_state],softmax_temp)
            next_action = egreedy_sample(mu,epsilon)
        else :
            next_action = sample(fixed_mu[env.current_state])

        actions[0] = next_action
        Q[0] = qvalues[env.current_state,actions[0]]
        states[0] = env.current_state

        while True :
            t += 1

            if t < T:
                action = actions[t % n]
                next_state, reward = env.take_action(action,verbose=False)
                total_reward += reward
                states[(t+1)%n] = next_state

                if env.is_terminal_state():
                    T = t+1
                    deltas[t % n] = reward - Q[t % n]
                else :
                    # select and store next action
                    if fixed_mu is False:
                        mu = normalize(qvalues[next_state],softmax_temp)
                        mu = egreedy_probs(mu, epsilon)
#                        next_action = egreedy_sample(mu,epsilon)
                    else :
                        mu = fixed_mu[next_state]
                    next_action = sample(mu)

                    actions[(t+1) % n] = next_action

                    #select and store sigma
                    next_sigma = get_sigma_func(env)
                    sigmas[(t+1) % n] = next_sigma

                    #Compute and store values
                    Q[(t+1) % n] = qvalues[next_state, next_action]
                    deltas[t % n] = reward + \
                                    gamma * next_sigma * Q[(t+1) % n] + \
                    gamma * (1 - next_sigma) * np.sum([policy[next_action][a]*qvalues[next_state,a] for a in range(env.num_actions)]) - Q[t % n]
                    pis[(t+1) % n] = policy[next_state][next_action]
                    ros[(t+1) % n] = pis[(t+1) % n] / mu[next_action]

            # tau is the time whose estimate is being updated
            tau = t - n #+ -1

            if tau >= 0:
                ro, E, G = 1, 1, Q[tau % n]

                for k in range(tau, min(tau+n-1,T-1)):
                    G = G + E * deltas[k % n]
                    E = gamma* E*((1-sigmas[(k+1)%n])* pis[(k+1)%n] + sigmas[(k+1)%n])
                    ro = ro * (1- sigmas[k % n] + sigmas[k % n]*ros[k % n])

                state_tau, action_tau = states[tau % n], actions[tau%n]
                qvalues[state_tau,action_tau] = qvalues[state_tau,action_tau] + alpha*ro*(G-qvalues[state_tau,action_tau])

                if fixed_pi is False:
                    policy[state_tau] = egreedy_probs(normalize(qvalues[state_tau],softmax_temp),epsilon*0.5)



            if env.is_terminal_state() or t > max_iter:
                if verbose :
                    if t > max_iter:
                        print("Max iteration reached.")
                    print('Episode done. Took '+str(t)+" iteration.")
                    print(qvalues)
                break


        if not evaluate_pi:
            stats.append((total_reward,t))
        else :
            pi_reward, pi_it_taken = evaluate(env,policy)
            stats.append((total_reward, t,pi_reward, pi_it_taken))
    print("Training done!")
    print('rewards')
    rewards = list(map(itemgetter(0), stats))
    print(rewards)
    print('iter_num')
    iter_taken = list(map(itemgetter(1), stats))
    print(iter_taken)
    if evaluate_pi :
        print('rewards')
        pi_rewards = list(map(itemgetter(2), stats))
        print(pi_rewards)
        print('iter_num')
        pi_iter_taken = list(map(itemgetter(3), stats))
        print(pi_iter_taken)

    if show_graph :
        print("showing graph")
        env.print_inf(qvalues, rewards, iter_taken, label)

    if evaluate_pi :
        return qvalues, rewards, iter_taken, (pi_rewards, pi_iter_taken)
    else :
        return qvalues, rewards, iter_taken

import numpy as np
from utils import normalize, egreedy_probs, egreedy_sample, sample
from operator import itemgetter



def Treebackup(env, n=2, fixed_pi=False, alpha=0.05, gamma=0.9, epsilon=5e-2, nb_episode=1000, max_iter=750, verbose=False,show_graph=False,label="Treebackup"):
    """
    n-step tree backup for estimating Q ~ q*, or Q ~ q_pi for a given pi
    Richard S. Sutton : Reinforcement Learning: An Introduction p.161
    """
    assert (gamma > 0 and gamma <= 1)
    assert (type(n) is int and n > 0)

    qvalues = np.ones((env.num_states, env.num_actions))
    qvalues[env.end_states[0]] = np.zeros((4,)) #Little fix because the end state is never updated

    stats = []
    if not fixed_pi is False :
        policy = fixed_pi

    for i in range(nb_episode):
        T, t, total_reward = float('inf'), -1, 0
        env.restart()

        Q = np.zeros((n,))
        states = np.zeros((n,),dtype=np.int)
        deltas = np.zeros((n,))
        actions = np.zeros((n,),dtype=np.int)
        pis = np.zeros((n,))

        if fixed_pi is False:
            policy = np.array([egreedy_probs(normalize(x),epsilon) for x in qvalues])

        pi = policy[env.current_state]
        actions[0] = sample(pi)

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
                    pi = policy[next_state]

                    deltas[t % n] = reward + \
                                    gamma * np.sum([pi[a] * qvalues[next_state, a] for a in range(env.num_actions)]) - \
                                    Q[t % n]
                    next_action = sample(pi)
                    actions[(t+1) % n] = next_action

                    #Compute and store values
                    Q[(t+1) % n] = qvalues[next_state, next_action]
                    pis[(t+1) % n] = pi[next_action]

            # tau is the time whose estimate is being updated
            tau = t - n #+ 1

            if tau >= 0:
                E, G = 1, Q[tau % n]

                for k in range(tau, min(tau+n-1,T-1)):
                    G = G + E * deltas[k % n]
                    E *= gamma*pis[(k+1)%n]

                state_tau, action_tau = states[tau % n], actions[tau%n]
              #  print(G,  qvalues[state_tau], action_tau)

                qvalues[state_tau,action_tau] = qvalues[state_tau,action_tau] + alpha*(G-qvalues[state_tau,action_tau])

                if fixed_pi is False:
                    policy[state_tau] = egreedy_probs(normalize(qvalues[state_tau]),epsilon)

            if env.is_terminal_state() or t > max_iter:
                if verbose :
                    if t > max_iter:
                        print("Max iteration reached.")
                    print('Episode done. Took '+str(t)+" iteration.")
                    print(qvalues)
                break

        stats.append((total_reward,t))
    print('rewards')
    rewards = list(map(itemgetter(0), stats))
    print(rewards)
    print('iter_num')
    iter_taken = list(map(itemgetter(1), stats))
    print(iter_taken)
    if show_graph :
        print("Training done! printing qvalues, showing graph")
        print(qvalues)
#        label = "%i-steps Treebackup"%n
        env.print_inf(qvalues,rewards, iter_taken,label)
    return qvalues, rewards, iter_taken
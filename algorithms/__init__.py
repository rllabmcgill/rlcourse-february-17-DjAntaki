import numpy as np
from utils import normalize, egreedy_sample
from operator import itemgetter

def Q_learning(env, gamma=0.9, alpha=0.05, nb_episode=1000, max_iter=750,verbose=False, show_graph=False, label="Q learning"):
    assert (gamma > 0 and gamma < 1)
    qvalue = np.ones((env.num_states, env.num_actions))
    qvalue[env.end_states[0]] = np.zeros((4,)) #Little fix because the end state is never updated

    stats = []

    for i in range(nb_episode):
        iter_count = 0
        total_reward = 0
        env.restart()
        while True:
            iter_count += 1
            current_state = env.current_state
            policy = normalize(qvalue[current_state])
            action = egreedy_sample(policy)
            next_state, reward = env.take_action(action, verbose=False)

            total_reward += reward

            qvalue[current_state, action] = qvalue[current_state, action] + \
                                            alpha * (
                                            reward + gamma * np.max(qvalue[next_state]) - qvalue[current_state, action])

            if env.is_terminal_state() or iter_count > max_iter:
                if verbose :
                    print('Episode done. Took ' + str(iter_count) + " iteration.")
                    print(qvalue)
                break

        stats.append((total_reward, iter_count))
    print('rewards')
    rewards = list(map(itemgetter(0), stats))
    print(rewards)
    print('iter_num')
    iter_taken = list(map(itemgetter(1), stats))
    print(iter_taken)

    if show_graph:
        env.print_inf(qvalue,rewards,iter_taken,label)
    return qvalue, rewards, iter_taken


def Q_learning2(env, gamma=0.9, alpha=0.05, nb_episode=1000, max_iter=750,verbose=False,show_graph=False,label="Expected Sarsa"):
    """ expected sarsa """
    assert (gamma > 0 and gamma < 1)
    qvalue = np.ones((env.num_states, env.num_actions))
    qvalue[env.end_states[0]] = np.zeros((4,)) #Little fix because the end state is never updated

    stats = []

    for i in range(nb_episode):
        iter_count = 0
        total_reward = 0
        env.restart()
        while True:
            iter_count += 1
            current_state = env.current_state
            policy = normalize(qvalue[current_state])
            action = egreedy_sample(policy)
            next_state, reward = env.take_action(action, verbose=False)
            total_reward += reward

            qvalue[current_state, action] = qvalue[current_state, action] + \
                                            alpha * (reward + gamma * np.sum(
                                                [policy[a] * qvalue[next_state, a] for a in range(env.num_actions)]) -
                                                     qvalue[current_state, action])

            if env.is_terminal_state() or iter_count > max_iter:
                if verbose is True :
                    print('Episode done. Took ' + str(iter_count) + " iteration.")
                    print(qvalue)
                    if iter_count > max_iter:
                        print("Max iteration reached. terminating.")
                break

        stats.append((total_reward, iter_count))

    print('rewards')
    rewards = list(map(itemgetter(0), stats))
    print(rewards)
    print('iter_num')
    iter_taken = list(map(itemgetter(1), stats))
    print(iter_taken)

    if show_graph:
        print("Training done! showing graph")
        env.print_inf(qvalue,rewards, iter_taken,label)
    return qvalue, rewards, iter_taken
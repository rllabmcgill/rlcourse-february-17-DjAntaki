import numpy as np

def kfold_decorator(func,k,show_graph=True):
    def kfold_func(env,*args,**kwargs):
        outputs = []
        for i in range(k):
            print("fold %i"%i)
            outputs.append(func(env,*args, **kwargs))

        qvalues, rewards, it_taken = list(map(np.array,zip(*outputs)))
        if show_graph :
            env.print_inf2(qvalues, rewards, it_taken,label=kwargs["label"])
        out = list(map(lambda x:np.average(x,axis=0),[qvalues, rewards, it_taken]))
        print('total it taken, avg reward last 100 episodes :',np.sum(out[2]),np.average(out[1][-100:]))
        return out
    return kfold_func

def evaluate(env,policy,nb_episode=1,max_iter=750,verbose=False):
    """evaluation of policy
    returns average reward and average number of iteration taken
    """
    rewards, iter_taken = [], []

    for i in range(nb_episode):
        env.restart()
        total_reward, t = 0, -1

        while True :
            t += 1
            action = sample(policy[env.current_state])
            _, reward = env.take_action(action)
            total_reward += reward

            if env.is_terminal_state() or t > max_iter:
                if verbose :
                    if t > max_iter:
                        print("Max iteration reached.")
                    print('Episode done. Took ' + str(t) + " iteration.")
                break

        rewards.append(total_reward)
        iter_taken.append(t)

    return np.average(total_reward),np.average(iter_taken)

def movingaverage(values, window):
    #taken from https://gordoncluster.wordpress.com/2014/02/13/python-numpy-how-to-generate-moving-averages-efficiently-part-2/
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

def sample(weights):
    return np.random.choice(range(len(weights)),p=weights)

def egreedy_sample(policy, epsilon=1e-2):
    """ returns argmax with prob (1-epsilon), else returns a random index"""
    if np.random.binomial(1,1-epsilon) == 1:
        return np.argmax(policy)
    else :
        return np.random.choice(range(len(policy)))

def onehot_enc(i,size):
    onehot = np.zeros((size,))
    onehot[i] = 1
    return onehot

def egreedy_probs(pi, epsilon=1e-2):
    """ pi is a 1-d array. returns probabilities"""
    return epsilon/(pi.size)*np.ones((pi.size,)) + (1-epsilon)*onehot_enc(np.argmax(pi),pi.size)

def esample_greedy_probs(pi,epsilon=1e-2):
    return epsilon*pi + (1-epsilon)*onehot_enc(np.argmax(pi),pi.size)

def eflatten_probs(pi,epsilon=1e-2):
    """  """
    return (1-epsilon)*pi + epsilon * uniform_probs((pi.size,))

def uniform_probs(shape):
    size = np.product(shape)
    return 1.0/size * np.ones(shape)

def softmax(x,temp=10):
    """A bit more stable softmax"""
    e_x = np.exp(x/temp)
    return e_x / e_x.sum(axis=0)

#def softmax(x):
#    """Not numerically stable"""
#    return np.exp(x) / np.sum(np.exp(x), axis=0)

def normalize(X,temp=10,do_softmax=True):
    if do_softmax is False :
        s = np.sum(X) #works only with non-negative qvalues

        if s == 0:
            return 1.0/len(X) * np.ones_like(X)
        return 1.0/s * X
    else :
        return softmax(X,temp)

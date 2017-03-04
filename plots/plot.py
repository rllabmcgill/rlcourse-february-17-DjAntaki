import numpy as np
from matplotlib import pyplot as plt
from utils import movingaverage


def make_plots(it_taken, rewards, lines_name, moving_avg=None, map_name="map#4",saveplot=True):
    """windyworld"""
    nb_epoch=len(it_taken[0])

    out = ""

    #    num_lines = len(lines_name)
#    colors = ['b-', 'g-','r-','g-','p-',][:lines_name]

    xpoints = list(range(nb_epoch))

    if not (moving_avg is None):
        rewards_1 = map(lambda x: movingaverage(x, moving_avg), rewards)
        it_taken_1 = map(lambda x: movingaverage(x, moving_avg), it_taken)
        xpoints = xpoints[moving_avg - 1:]
    else :
        rewards_1 = rewards
        it_taken_1 =rewards

    fig = plt.figure()
    plt.suptitle("Q(sigma) variants performance on "+map_name)
    ax = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)

    #in function of episode completed

    for nb_it, reward, algorithm_name in zip(it_taken_1, rewards_1, lines_name):#,colors):
        ax.plot(xpoints, nb_it, label=algorithm_name)
        ax2.plot(xpoints, reward, label=algorithm_name)

    ax.set_title('# of iteration taken in fct. of episodes completed')
    ax.set_ylabel("# iteration taken")
   # ax.set_xlabel("# episodes completed")
    legend = ax.legend(loc='upper right', shadow=True)

    ax2.set_ylabel("Reward")
    ax2.set_title('total reward in fct. of episodes completed')
  #  ax2.set_xlabel("# episodes completed")
    legend = ax2.legend(loc='lower right', shadow=True)
    #plt.show()

    #in function of episode taken

    ax3 = fig.add_subplot(3, 1, 3)

    for nb_it, reward, algorithm_name in zip(it_taken, rewards, lines_name):  # ,colors):
        xpoints2 = np.zeros(len(nb_it))
        xpoints2[0] = nb_it[0]
        for i in range(1,len(nb_it)):
            xpoints2[i] = nb_it[i] + xpoints2[i-1]
        ax3.plot(xpoints2, reward, label=algorithm_name)
        print(algorithm_name,xpoints2[-1], np.average(reward[-100:]))
        out += " & ".join([algorithm_name, str(round(xpoints2[-1],2)), str(round(np.average(reward[-100:]), 2))]) + "\\\\ \n"


    ax3.set_ylabel("Reward")
    ax3.set_title('total reward in fct. of iteration completed')
    #ax.set_xlabel("# iteration completed")
    legend = ax3.legend(loc='lower right', shadow=True)
    if saveplot:
        lbl = "qsigma_" + map_name.replace("#", '')
        plt.savefig(lbl + '.png', bbox_inches='tight')
    else:
        print(out)
        plt.show()



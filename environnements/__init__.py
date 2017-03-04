import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from utils import movingaverage

class Env:
    def print_inf(self, qvalues, rewards, it_taken, label):
        """
        :param qvalues:
        :param rewards:
        :param it_taken:
        :param pi_reward: is either None or a tuple containning the rewards and the iteration takens for the pi policy. Giving this input implies off-policy learning
        :param label:
        :return:
        """
        a, b = self.grid.shape
        zvals = np.array(
            [[np.max(qvalues[self.state_to_id[(i, j)]]) if (i, j) in self.state_space else float('nan') for i in
              range(a)] for j
             in range(b)])

        fig = plt.figure()
        cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
                                                             ['black', 'red'],
                                                             256)

        if True :
            ax = plt.subplot(2,2,1)

            img2 = ax.imshow(np.flipud(zvals.T), interpolation='nearest',
                              cmap=cmap2,
                              origin='lower')
            ax.set_yticks(np.arange(0, 4, 1.0))
            fig.colorbar(img2, cmap=cmap2)
            ax.set_title("max qvalue")
        else :
            img2 = plt.imshow(np.flipud(zvals.T), interpolation='nearest',
                              cmap=cmap2,
                              origin='lower')

            plt.yticks(np.arange(0, 4, 1.0))
            plt.colorbar(img2, cmap=cmap2)
            plt.title(label + " - max qvalue by state")
            plt.show()

        zvals = np.array(
            [[np.argmax(qvalues[self.state_to_id[(i, j)]]) if (i, j) in self.state_space else float('nan') for i in
              range(a)] for j
             in range(b)])

        if True :
            ax2 = plt.subplot(2,2,2)
            img2 = ax2.imshow(np.flipud(zvals.T), interpolation='nearest',
                              origin='lower')
            #fig.colorbar(img2, cmap=cmap2)
            ax2.set_yticks(np.arange(0, 4, 1.0))
            ax2.set_title("greedy policy")

        else :
            img2 = plt.imshow(np.flipud(zvals.T), interpolation='nearest',
                              origin='lower')
            #plt.colorbar(img2,cmap=cmap2)
            plt.yticks(np.arange(0, 4, 1.0))
            plt.title(label + " algorithm - greedy policy")
            plt.show()

        xpoints = xrange(len(rewards))

        if True :
            ax3 = plt.subplot(2,2,3)
            ax3.plot(xpoints, rewards, label=label+" - rewards")
            ax3.set_title('R in fct. of episodes done')
            ax3.set_ylabel("Reward")
            ax3.set_xlabel("# of episodes done")

            ax4 = plt.subplot(2,2,4)
            ax4.plot(xpoints, it_taken, label=label+" - iteration taken")
            ax4.set_title('# of it. taken in fct. of episodes done')
            ax4.set_ylabel("# iteration taken")
            ax4.set_xlabel("# of episodes completed")



        plt.show()

    def print_inf2(self, qvalues, rewards, it_taken, label, avg_window_size=10,assembled_plot=True,pi_stats=None,saveplot=True):
        """ for multiples folds """
        a, b = self.grid.shape
        map_name = self.map_name
#        std_qvalues = np.std(qvalues, axis=0)
        avg_qvalues = np.average(qvalues,axis=0)

        if not(avg_window_size is None):
            assert type(avg_window_size) is int


        zvals = np.array(
            [[np.max(avg_qvalues[self.state_to_id[(i, j)]]) if (i, j) in self.state_space else float('nan') for i in
              range(a)] for j
             in range(b)])

        fig = plt.figure()
        cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
                                                             ['black', 'red'],
                                                             256)

        if assembled_plot :
            ax = plt.subplot(2,2,1)

            img2 = ax.imshow(np.flipud(zvals.T), interpolation='nearest',
                              cmap=cmap2,
                              origin='lower')
            ax.set_yticks(np.arange(0, a, 1.0))
            fig.colorbar(img2, cmap=cmap2)
            ax.set_title("max qvalue")
        else :
            img2 = plt.imshow(np.flipud(zvals.T), interpolation='nearest',
                              cmap=cmap2,
                              origin='lower')

            plt.yticks(np.arange(0, a, 1.0))
            plt.colorbar(img2, cmap=cmap2)
            plt.title(label + " - max qvalue by state")
            plt.show()

        greedy_policy = np.array(
            [[np.argmax(avg_qvalues[self.state_to_id[(i, j)]]) if (i, j) in self.state_space else float('nan') for i in
              range(a)] for j
             in range(b)])

        if assembled_plot :
            ax2 = plt.subplot(2,2,2)
            img2 = ax2.imshow(np.flipud(greedy_policy.T), interpolation='nearest',
                              origin='lower')
            ax2.set_yticks(np.arange(0, a, 1.0))
            ax2.set_title("greedy policy")

        else :
            img2 = plt.imshow(np.flipud(greedy_policy.T), interpolation='nearest',
                              origin='lower')
            plt.yticks(np.arange(0, a, 1.0))
            plt.title(label + " algorithm - greedy policy")
            plt.show()

#        zvals = np.array(
#            [[std_qvalues[self.state_to_id[(i, j)]][int(greedy_policy[i,j])] if (i, j) in self.state_space else float('nan') for i in
#              range(a)] for j
#             in range(b)])

#        if assembled_plot :
#            ax3 = plt.subplot(2,3,3)
#            img2 = ax3.imshow(np.flipud(zvals.T), interpolation='nearest',
#                              origin='lower')
#            ax3.set_yticks(np.arange(0, a, 1.0))
#            ax3.set_title("qvalue variance")


        xpoints = range(len(rewards[0]))

        if assembled_plot :
            ax5 = plt.subplot(2,2,3)
            ax6 = plt.subplot(2,2,4)

            if not (avg_window_size is None):
                rewards = map(lambda x:movingaverage(x,avg_window_size),rewards)
                it_taken = map(lambda x:movingaverage(x,avg_window_size),it_taken)
                xpoints = xpoints[avg_window_size-1:]

            if not (pi_stats is None):
                pi_rewards = map(lambda x:movingaverage(x,avg_window_size),pi_stats[0])
                pi_it_taken = map(lambda x:movingaverage(x,avg_window_size),pi_stats[1])
                avg_pi_rewards = np.average(pi_rewards,axis=0)
                avg_pi_it_taken = np.average(pi_it_taken,axis=0)


            avg_rewards = np.average(rewards,axis=0)
            avg_it_taken = np.average(it_taken,axis=0)

            if not (pi_stats is None):
                it = -1
                for r,pr,t,pt in zip(rewards,pi_rewards,it_taken,pi_it_taken):
                    it += 1
                    print(it)
                    if any(r > 0):
                        print("r")
                    if any(pr>0):
                        print(pr)
                        print("pr")
                    if any(t) <= 0 :
                        print("t")
                    if any(pt) <= 0 :
                        print("pt")
                    ax5.plot(xpoints,r,'k-',alpha=0.2)
                    ax5.plot(xpoints,pr,'b-',alpha=0.2)
                    ax6.plot(xpoints,t,'k-', alpha=0.2)
                    ax6.plot(xpoints,pt,'b-',alpha=0.2)
            else :
                for r,t in zip(rewards,it_taken):
                    ax5.plot(xpoints,r,'k-',alpha=0.2)
                    ax6.plot(xpoints, t,'k-', alpha=0.2)


            ax5.plot(xpoints, avg_rewards,'k-')
            ax6.plot(xpoints, avg_it_taken,'k-')

            if not (pi_stats is None) :
                ax5.plot(xpoints, avg_pi_rewards,'b-')
                ax6.plot(xpoints, avg_pi_it_taken,'b-')

            ax5.set_title('R in fct. of episodes done')
            ax5.set_ylabel("Reward")
            ax5.set_xlabel("# of episodes done")

            ax6.set_title('# of it. taken in fct. of episodes done')
            ax6.set_ylabel("# iteration taken")
            ax6.set_xlabel("# of episodes completed")

        plt.suptitle(label + " - "+map_name)
        if saveplot:
            lbl = label.replace(' ','') + "_"+ map_name.replace("#",'')
            plt.savefig(lbl+'.png', bbox_inches='tight')
        else :
            plt.show()

    def is_terminal_state(self):
        return self.current_state in self.end_states
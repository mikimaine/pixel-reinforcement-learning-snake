import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animate(i, scores, avg_rewards, losses, epsilons):
    plt.clf()

    plt.subplot(411)
    plt.plot(scores, label='Score per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.legend()

    plt.subplot(412)
    plt.plot(avg_rewards, label='Average Reward per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.legend()

    plt.subplot(413)
    plt.plot(epsilons, label='Epsilon Decay')
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon')
    plt.legend()

    plt.subplot(414)
    plt.plot(losses, label='Loss')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()


def setup_plot(scores, avg_rewards, losses, epsilons):
    fig = plt.figure()
    ani = animation.FuncAnimation(fig, animate, fargs=(scores, avg_rewards, losses, epsilons), interval=1000)
    plt.show(block=False)
    return ani

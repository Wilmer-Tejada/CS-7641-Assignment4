import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
from hiive import mdptoolbox
import time
import numpy as np
import matplotlib.pyplot as plt

# # VALUE ITERATION
# QI = mdptoolbox.mdp.QLearning(transitions=P, reward=R,gamma =.9, epsilon=0.01)
# QI.run()

# QI.V # Value Function
# QI.policy # optimal policy
# QI.iter # number of iterations
# QI.time # used CPU time

#########################################
if __name__ == '__main__':

    # random_map = generate_random_map(size=8, p=0.98)
    # P, R = mdptoolbox.example.openai("FrozenLake-v1", desc=random_map)
    P, R = mdptoolbox.example.forest(2000)

    # env_name  = "FrozenLake8x8-v1"
    # env = gym.make(env_name)
    # env = env.unwrapped

    times = []
    gammas = []
    iterations = []
    listscore = []
    rewards_list = []


    QI = mdptoolbox.mdp.QLearning(transitions=P, reward=R, gamma=.99, epsilon_decay=0.99999, alpha_decay=.99999)
    Qs = []
    average_per_episode = []
    for episode in range(10000):
        run_stats = QI.run()
        Qs.append(QI.Q)
        average_per_episode.append(np.mean([x['Mean V'] for x in run_stats]))

    plt.close()
    plt.plot(average_per_episode)
    plt.title("Q Learning - Score by Episode")
    plt.ylabel("Score (Mean V)")
    plt.xlabel("Episode")
    plt.savefig("Assignment4/ForestCharts/QI Gammas and Scores.png")



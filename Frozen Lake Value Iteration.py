
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
from hiive import mdptoolbox
import time
import numpy as np
import matplotlib.pyplot as plt

# # VALUE ITERATION
# VI = mdptoolbox.mdp.ValueIteration(transitions=P, reward=R,gamma =.9, epsilon=0.01)
# VI.run()

# VI.V # Value Function
# VI.policy # optimal policy
# VI.iter # number of iterations
# VI.time # used CPU time

#########################################
if __name__ == '__main__':
    np.random.seed(0)
    # random_map = generate_random_map(size=8, p=0.98)
    # P, R = mdptoolbox.example.openai("FrozenLake-v1", desc=random_map)
    P, R = mdptoolbox.example.openai("FrozenLake8x8-v1")

    env_name  = "FrozenLake8x8-v1"
    env = gym.make(env_name)
    env = env.unwrapped

    times = []
    gammas = []
    iterations = []
    listscore = []
    rewards_list = []

    gammas = np.linspace(0.9,1, 10)

    for i in gammas:
        start = time.time()
        VI = mdptoolbox.mdp.ValueIteration(transitions=P, reward=R, gamma=i, epsilon=0.01, max_iter=10000)
        VI.run()
        optimal_policy = VI.policy
        scores = np.mean(VI.V)
        rewards = np.max(VI.V)

        end = time.time()
        # gammas.append(i)
        listscore.append(scores)
        iterations.append(VI.iter) # number of iterations
        times.append(end - start)
        rewards_list.append(rewards)


    # plt.plot(gammas, times)
    # plt.title('Policy Iteration - Time of Execution')
    # plt.xlabel('Gammas')
    # plt.ylabel('Time of execution')
    # # plt.savefig()
    # plt.show()

    # plt.plot(gammas, iterations)
    # plt.title('Policy Iteration - Iterations')
    # plt.xlabel('Gammas')
    # plt.ylabel('Iterations')
    # plt.show()

    plt.close()
    plt.plot(gammas, listscore)
    plt.title('Value Iteration - Scores')
    plt.xlabel('Gammas')
    plt.ylabel('Scores (Mean V)')
    # plt.show()
    plt.savefig("Assignment4/FrozenLakeCharts/VI Gammas and Scores.png")

    plt.close()
    plt.plot(iterations, rewards_list)
    plt.title('Value Iteration - Rewards')
    plt.xlabel('Iterations')
    plt.ylabel('Rewards (Max V)')
    # plt.show()
    plt.savefig("Assignment4/FrozenLakeCharts/VI Rewards and Iterations.png")

    #
    # plt.plot(iterations, listscore)
    # plt.title('Value Iteration - Rewards by Scores')
    # plt.xlabel('Iterations')
    # plt.ylabel('Scores')
    # plt.show()


    # BEST MODEL

    P, R = mdptoolbox.example.openai("FrozenLake8x8-v1")
    VI = mdptoolbox.mdp.ValueIteration(transitions=P, reward=R, gamma=.99, epsilon=0.01, max_iter=10000)
    # VI.setVerbose()
    stats = VI.run()
    optimal_policy = VI.policy
    steps_list = []
    TEST = np.array(optimal_policy, dtype=float)
    episode_list = []
    e = 0
    for i_episode in range(1000):
        episode_list.append(i_episode)
        c = env.reset()
        steps = 0
        for t in range(10000):
            steps += 1
            c, reward, done, info = env.step(TEST[c])
            if done:
                if reward == 1:
                    e += 1
                    steps_list.append(steps)

                break
    print(" agent succeeded to reach goal {} out of 1000 Episodes using this policy ".format(e + 1))
    print('----------------------------------------------')
    print('You took an average of {:.0f} steps to get to the goal'.format(np.mean(steps_list)))

    plt.close()
    plt.plot(steps_list)
    plt.plot(np.mean(steps_list))
    plt.title("Steps to converge")
    plt.show()


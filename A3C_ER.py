import collections
import random
import time
import warnings

import gym  # openai gym library
import torch
import torch.multiprocessing as mp  # multi processing
import torch.nn as nn  # Linear
import torch.nn.functional as F  # relu, softmax
import torch.optim as optim  # Adam Optimizer
from torch.distributions import Normal

warnings.filterwarnings("ignore")

import numpy as np
from matplotlib import pyplot as plt  # ##for plot

import wandb

# Hyperparameters
n_train_processes = 3
learning_rate = 0.0002
update_interval = 5
gamma = 0.98
max_train_ep = 300
max_test_ep = 400
logging = False
buffer_limit = 100
initial_exp = 30


torch.autograd.set_detect_anomaly(True)


class ReplayBuffer:
    def __init__(self) -> None:
        self.buffer = collections.deque(maxlen=buffer_limit)
        # self.buffer = []

    def put(self, transition):
        self.buffer.append(transition)

    def make_batch(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, p_lst, s_prime_lst, done_lst = [], [], [], [], [], []

        for transition in mini_batch:
            s, a, r, log_prob, s_prime, done = transition

            s_lst.append(s)
            a_lst.append([a])
            p_lst.append(log_prob)
            r_lst.append(r / 100.0)
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])

        s_batch, a_batch, r_batch, p_batch, s_prime_batch, done_batch = (
            torch.tensor(s_lst, dtype=torch.float),
            torch.tensor(a_lst),
            torch.tensor(r_lst, dtype=torch.float),
            torch.stack(p_lst),
            torch.tensor(s_prime_lst, dtype=torch.float),
            torch.tensor(done_lst, dtype=torch.float),
        )

        return s_batch, a_batch, r_batch, p_batch, s_prime_batch, done_batch

    def size(self):
        return len(self.buffer)


# This class is equivalent to Actor-Critic. (pi, v)
class ActorCritic(
    nn.Module
):  # ActorCritic Class - Created by inheriting nn.Module (provided by Pytorch) Class.
    def __init__(
        self,
    ):  # constructor - initializer(__init__): Object creation and variable initialization.
        super(
            ActorCritic, self
        ).__init__()  # Calling the constructor of the inherited parent Class(nn.Module).
        self.fc1 = nn.Linear(11, 256)  # Fully Connected: input   4 --> output 256
        self.fc_pi = nn.Linear(256, 3)  # Fully Connected: input 256 --> output   2
        self.std = nn.Linear(256, 3)
        self.fc_v = nn.Linear(256, 1)  # Fully Connected: input 256 --> output   1

        self.data = []

    def pi(
        self, x, softmax_dim=0
    ):  # In the case of batch processing, softmax_dim becomes 1. (default 0)
        x = F.relu(self.fc1(x))

        std = F.relu(self.std(x))
        std = torch.clamp(std, min=-20, max=2)
        std = std.exp()

        mean = self.fc_pi(x)

        dist = Normal(mean, std)

        return dist

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, p_lst, s_prime_lst, done_lst = [], [], [], [], [], []

        for transition in self.data:
            s, a, r, log_prob, s_prime, done = transition

            s_lst.append(s)
            a_lst.append([a])
            p_lst.append(log_prob)
            r_lst.append(r / 100.0)
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])

            s_batch, a_batch, r_batch, p_batch, s_prime_batch, done_batch = (
                torch.tensor(s_lst, dtype=torch.float),
                torch.tensor(a_lst),
                torch.tensor(r_lst, dtype=torch.float),
                torch.stack(p_lst),
                torch.tensor(s_prime_lst, dtype=torch.float),
                torch.tensor(done_lst, dtype=torch.float),
            )

        self.data = []
        return s_batch, a_batch, r_batch, p_batch, s_prime_batch, done_batch


def train(global_model, rank):  # Called by 3(n_train_processes) agents independently.
    count = 0
    memory = ReplayBuffer()
    local_model = (
        ActorCritic()
    )  # Call the ActorCritic.__init__() --> Creating an local_model Object
    local_model.load_state_dict(
        global_model.state_dict()
    )  # Copy the global_model network weights & biases to local_model # local_model=global_model

    optimizer = optim.Adam(
        global_model.parameters(), lr=learning_rate
    )  # The optimizer updates the parameters of 'global_model'.

    env = gym.make("Hopper-v3")  # Create 'CartPole-v1' environment.

    for n_epi in range(max_train_ep):  # max_train_ep = 300
        done = False  # done becomes True when the episode ends.
        s = (
            env.reset()
        )  # Reset Environment - s = [0.02482228  0.00863265 -0.0270073  -0.01102263]

        while not done:  # CartPole-v1 forced to terminates at 500 steps.

            s_lst, a_lst, r_lst, p_lst = [], [], [], []

            for t in range(
                update_interval
            ):  # Collect data during 5(update_interval) steps and proceed with training.
                dist = local_model.pi(torch.from_numpy(s).float())
                a = dist.sample()
                log_prob = dist.log_prob(a).sum()
                a = F.tanh(a)  # [-1,1]
                a = a.reshape(-1).numpy()

                s_prime, r, done, info = env.step(a)

                memory.put((s, a, r, log_prob, s_prime, done))

                # count += a.shape[0]

                s = s_prime
                if done:
                    break

            if memory.size() > initial_exp:

                (
                    s_batch,
                    a_batch,
                    r_batch,
                    p_batch,
                    s_prime_batch,
                    done_batch,
                ) = memory.make_batch(20)

                r_lst = list(r_batch.detach().numpy())

                s_final = torch.tensor(
                    s_prime, dtype=torch.float
                )  # numpy array to tensor - s_final[4]
                R = (
                    0.0 if done else local_model.v(s_final).item()
                )  # .item() is to change tensor to python float type.
                td_target_lst = []
                for reward in r_lst[
                    ::-1
                ]:  # r_lst[start,end,step(-1)] ==> 5(update_interval):0:-1
                    R = gamma * R + reward
                    td_target_lst.append([R])
                td_target_lst.reverse()

                td_target_batch = torch.tensor(td_target_lst)

                advantage = td_target_batch - local_model.v(s_batch)

                loss = torch.sum(
                    -p_batch * advantage.detach().reshape(-1).float()
                ) + F.mse_loss(
                    local_model.v(s_batch), td_target_batch.detach().float()
                )  # This is equivalent to Actor-Critic.

                optimizer.zero_grad()
                loss.mean().backward(
                    retain_graph=True
                )  # Backpropagation - gradient calculation

                for global_param, local_param in zip(
                    global_model.parameters(), local_model.parameters()
                ):
                    global_param.grad = (
                        local_param.grad
                    )  # local_param.grad -> A variable that stored the calculated gradient values.

                optimizer.step()  # weights & biases update
                for name, param in local_model.named_parameters():
                    param.data = global_model.state_dict()[name]
                    """
                    alterntive to get over the following error:
                    RuntimeError: one of the variables needed for gradient
                    computation has been modified by an inplace operation
                    """

    env.close()

    print("Training process {} reached maximum episode.".format(rank))


def test(global_model):
    env = gym.make("Hopper-v3")  # Create 'CartPole-v1' environment.
    score = 0.0
    print_interval = 20

    if logging:
        wandb.init(project="A3C")

    render = False  # for rendering

    reward_means = []  ###for plot
    reward_stds = []  ###for plot
    rwd_list = []  ###for plot

    for n_epi in range(max_test_ep):  # max_test_ep = 400
        done = False  # done becomes True when the episode ends.
        s = env.reset()  # Reset Environment

        rwd_sum = 0.0  # Sum of rewards for each episode

        while not done:
            dist = global_model.pi(torch.from_numpy(s).float())
            a = dist.sample()
            a = F.tanh(a)
            a = a.reshape(-1).numpy()

            s_prime, r, done, info = env.step(a)
            s = s_prime

            score += r  # Sum of rewards for max_test_ep episode
            rwd_sum += r  # Sum of rewards for each episode

            if render:  # for rendering
                env.render()

        rwd_list.append(rwd_sum)

        if n_epi % print_interval == 0 and n_epi != 0:
            print(
                "# of episode :{:4d}, avg score : {:7.1f}, std : {:7.1f}".format(
                    n_epi, score / print_interval, np.std(rwd_list)
                )
            )

            if logging:
                wandb.log({"n_episode": n_epi, "score": score / print_interval})
            reward_means.append(score / print_interval)  ###for plot
            reward_stds.append(np.std(rwd_list))  ###for plot
            rwd_list = []

            if abs(score / print_interval) > 450:  # for rendering
                render = False
            score = 0.0
            time.sleep(1)  # 1 second.

    env.close()
    if logging:
        wandb.finish()

    print_reward(
        reward_means, reward_stds, print_interval, "A3C", "g"
    )  # label name, color='r','g','b','c','m','y','k','w'


def print_reward(rwds, stds, eval_interval, label_name, color):  # for plot

    x = np.arange(len(rwds)) + 1
    x = (
        x * eval_interval
    )  # The test is run every eval_interval, and the mean and variance of the reward are stored.

    stds = np.array(stds)  # list to array
    rwds = np.array(rwds)

    y1 = rwds + stds  # for plot variance
    y2 = rwds - stds

    plt.plot(x, rwds, color=color, label=label_name)  # plot average reward
    plt.fill_between(x, y1, y2, color=color, alpha=0.1)  # plot variance

    plt.xlabel("Environment Episodes", fontsize=15)
    plt.ylabel("Average Reward", fontsize=15)

    plt.legend(loc="best", fontsize=15, frameon=False)  # frameon=False: no frame border
    # plt.legend(loc=(0.7,0.1), ncol=2, fontsize=15, frameon=False) #position 0,0 ~ 1,1 (x, y)

    plt.grid(color="w", linestyle="-", linewidth=2, alpha=0.3)  # grid

    plt.tick_params(axis="x", labelsize=10, color="w")
    plt.tick_params(axis="y", labelsize=10, color="w")

    ax = plt.gca()
    ax.set_facecolor("#EAEAF2")  # background color

    # Plot border invisible
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # plt.savefig("plot.png")
    plt.plot()


if __name__ == "__main__":
    global_model = ActorCritic()
    global_model.share_memory()  # Move 'global_model' to shared memory to share data between multiple processes.

    memory = ReplayBuffer()

    processes = []
    # Create 3(n_train_processes) processes for training and 1 process for testing.
    for rank in range(n_train_processes + 1):  # + 1 for test process
        if rank == 0:
            p = mp.Process(
                target=test, args=(global_model,)
            )  # Create a test processor.
        else:
            p = mp.Process(
                target=train,
                args=(
                    global_model,
                    rank,
                ),
            )  # Create training processor.

        p.start()  # Process Start
        processes.append(p)

    for p in processes:
        p.join()  # Wait for all processes to terminate.

import argparse
import collections
import random

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import wandb

learning_rate = 0.0002
gamma = 0.98
n_rollout = 10


class ReplayBuffer:
    def __init__(self, args) -> None:
        self.buffer = collections.deque(maxlen=args.buffer_limit)

    def put(self, transition):  # TODO: use instead of put data
        self.buffer.append(transition)

    def sample(self, n):  # TODO: use instead of make batch
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return (
            torch.tensor(s_lst, dtype=torch.float, device=device),
            torch.tensor(a_lst, device=device),
            torch.tensor(r_lst, device=device),
            torch.tensor(s_prime_lst, dtype=torch.float, device=device),
            torch.tensor(done_mask_lst, device=device),
        )

    def size(self):
        return len(self.buffer)


class ActorCritic(nn.Module):
    def __init__(self) -> None:
        super(ActorCritic, self).__init__()
        # self.data = []

        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v


def train_net(model, memory, optimizer, args):
    s, a, r, s_prime, done = memory.sample()
    td_target = r + gamma * model.v(s_prime) * done
    delta = td_target - model.v(s)
    pi = model.pi(s, softmax_dim=1)
    pi_a = pi.gather(1, a)
    loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(
        model.v(s), td_target.detach()
    )

    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()


def main(args):

    global device

    if args.wandb:
        # print("Wandb running!")
        wandb.init(project="cartpole-gym")
        # wandb.config.update(args)

    device = "cpu" if args.gpu == -1 else f"cuda:{args.gpu}"

    env = gym.make("CartPole-v1")
    memory = ReplayBuffer(args)
    model = ActorCritic(args).to(device)
    print_interval = 20
    score = 0.0
    max_episode = 10000
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    render = False

    for n_epi in range(max_episode):
        done = False
        s = env.reset()

        while not done:
            for t in range(n_rollout):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step(a)
                memory.put_data((s, a, r, s_prime, done))

                s = s_prime
                score += r

                if render:
                    env.render()

                if done:
                    break

            train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print(
                "# of episode: {}, avg score: {:.1f}".format(
                    n_epi, score / print_interval
                )
            )
            wandb.log({"n_episode": n_epi, "score": score / print_interval})
            if abs(score / print_interval) > 350:
                render = True
            score = 0.0
    env.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="cartpole-gym")
    parser.add_argument(
        "--gpu", type=int, default=0, help="gpu device num. -1 is for cpu"
    )
    parser.add_argument("--wandb", type=bool, default=False)

    args = parser.parse_args()

    main(args)

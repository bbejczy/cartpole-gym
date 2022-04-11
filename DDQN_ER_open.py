import argparse
import collections
import random

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import wandb

"""
Provided Hyperparameters

learning_rate = 0.0005
gamma = 0.98
max_episode = 20000
buffer_limit = 10000
batch_size = 32  # TODO: define batch size
initial_exp = 5000  # TODO: define initial experience

"""


class ReplayBuffer:
    def __init__(self, args) -> None:
        self.buffer = collections.deque(maxlen=args.buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
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
            torch.tensor(s_lst, dtype=torch.float).to(device),
            torch.tensor(a_lst).to(device),
            torch.tensor(r_lst).to(device),
            torch.tensor(s_prime_lst, dtype=torch.float).to(device),
            torch.tensor(done_mask_lst).to(device),
        )

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()

        self.input_dim = 4
        self.output_dim = 2

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.model(x)
        return x

    def sample_action(self, obs, epsilon):

        q_out = self.forward(obs)
        r_value = random.random()

        if r_value < epsilon:
            return random.randint(0, 1)
        else:
            return q_out.argmax().item()


def train(q, q_target, memory, optimizer, args):

    for i in range(10):

        s, a, r, s_prime, done_mask = memory.sample(args.batch_size)
        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + args.gamma * max_q_prime * done_mask

        loss = F.smooth_l1_loss(q_a, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main(args):

    global device

    device = "cpu" if args.gpu == -1 else f"cuda:{args.gpu}"

    if args.wandb:
        wandb.init(project="cartpole-gym", monitor_gym=args.monitor_gym)
        wandb.config.update(args)

    env = gym.make("CartPole-v1")

    q = Qnet().to(device)
    q_tagret = Qnet().to(device)
    q_tagret.load_state_dict(q.state_dict())

    memory = ReplayBuffer(args)
    print_interval = 20
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=args.learning_rate)

    for n_epi in range(args.max_episode):

        epsilon = max(0.01, 0.1 - 0.01 * (n_epi / 200))
        s = env.reset()
        done = False

        while not done:

            a = q.sample_action(torch.from_numpy(s).float(), epsilon)

            s_prime, r, done, info = env.step(a)

            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r / 100.0, s_prime, done_mask))
            s = s_prime

            score += r

        if memory.size() > args.initial_exp:
            train(q, q_tagret, memory, optimizer, args)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_tagret.load_state_dict(q.state_dict())
            print(
                "n_episode : {}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                    n_epi, score / print_interval, memory.size(), epsilon * 100
                )
            )
            wandb.log(
                {
                    "n_episode": n_epi,
                    "score": score / print_interval,
                    "eps": epsilon * 100,
                }
            )

            score = 0.0

    env.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="cartpole-gym")
    parser.add_argument(
        "--gpu", type=int, default=0, help="gpu device num. -1 is for cpu"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--initial_exp", type=int, default=500)
    parser.add_argument("--max_episode", type=int, default=20000)
    parser.add_argument("--buffer_limit", type=int, default=10000)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--monitor_gym", type=bool, default=False)

    args = parser.parse_args()

    main(args)

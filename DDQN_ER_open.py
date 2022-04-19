import collections
import random
import re

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
max_episode = 2000
buffer_limit = 10000
batch_size = 64  # TODO: define batch size
initial_exp = 5000  # TODO: define initial experience



class ReplayBuffer:
    def __init__(self) -> None:
        self.buffer = collections.deque(maxlen=buffer_limit)

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
            torch.tensor(s_lst, dtype=torch.float, device=device),
            torch.tensor(a_lst, device=device),
            torch.tensor(r_lst, device=device),
            torch.tensor(s_prime_lst, dtype=torch.float, device=device),
            torch.tensor(done_mask_lst, device=device),
        )

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        # TODO: design network architecture
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # TODO: design forward pass
        x = self.fc2(x)
        return x

    def sample_action(self, obs, epsilon):

        q_out = self.forward(obs)
        r_value = random.random()

        if r_value < epsilon:
            return random.randint(0, 1)
        else:
            return q_out.argmax().item()


def train(q, q_target, memory, optimizer):

    for i in range(10):

        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask

        loss = F.smooth_l1_loss(q_a, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():

    global device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')


    env = gym.make("CartPole-v1")

    q = Qnet().to(device)
    q_tagret = Qnet().to(device)
    q_tagret.load_state_dict(q.state_dict())

    memory = ReplayBuffer()
    print_interval = 20
    score = 0.0
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(max_episode):

        epsilon = max(0.01, 0.1 - 0.01 * (n_epi / 200))
        s = env.reset()
        done = False

        while not done:

            a = q.sample_action(torch.from_numpy(s).float().to(device), epsilon)

            s_prime, r, done, info = env.step(a)

            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r / 100.0, s_prime, done_mask))
            s = s_prime

            score += r

        if memory.size() > initial_exp:
            train(q, q_tagret, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_tagret.load_state_dict(q.state_dict())
            print(
                "n_episode : {}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                    n_epi, score / print_interval, memory.size(), epsilon * 100
                )
            )

            score = 0.0

    env.close()


if __name__ == "__main__":
    main()

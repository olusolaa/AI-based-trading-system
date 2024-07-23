import random as rand
import numpy as np


class CircularBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = np.zeros((capacity, 4))
        self.index = 0
        self.size = 0

    def add(self, experience):
        self.buffer[self.index] = experience
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n):
        indices = np.random.choice(self.size, n, replace=False)
        return self.buffer[indices]


class QLearner(object):
    def __init__(self, num_states=100, num_actions=4, alpha=0.2, gamma=0.9, rar=0.5, radr=0.99, dyna=0, verbose=False):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.verbose = verbose

        self.Q = np.zeros((num_states, num_actions))
        self.experience = CircularBuffer(10000)

        self.s = 0
        self.a = 0

    def author(self):
        return "oalao30"  # Replace with your Georgia Tech username

    def querysetstate(self, s):
        self.s = s
        action = self._choose_action(s)
        self.a = action
        if self.verbose:
            print(f"s = {s}, a = {action}")
        return action

    def query(self, s_prime, r):
        # Q-Learning update
        self._update_q_table(self.s, self.a, s_prime, r)

        # Store experience for Dyna-Q
        if self.dyna > 0:
            self.experience.add((self.s, self.a, s_prime, r))

        # Dyna-Q updates
        if self.dyna > 0:
            self._dyna_q_updates()

        # Decay random action rate
        self.rar *= self.radr

        # Select next action
        self.s = s_prime
        action = self._choose_action(s_prime)
        self.a = action

        if self.verbose:
            print(f"s = {s_prime}, a = {action}, r = {r}")
        return action

    def _update_q_table(self, s, a, s_prime, r):
        max_q_s_prime = np.max(self.Q[s_prime])
        self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + self.alpha * (r + self.gamma * max_q_s_prime)

    def _choose_action(self, state):
        if rand.random() < self.rar:
            return rand.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.Q[state])

    def _dyna_q_updates(self):
        num_samples = min(self.dyna, self.experience.size)
        sampled_experiences = self.experience.sample(num_samples)

        s_dyna = sampled_experiences[:, 0].astype(int)
        a_dyna = sampled_experiences[:, 1].astype(int)
        s_prime_dyna = sampled_experiences[:, 2].astype(int)
        r_dyna = sampled_experiences[:, 3]

        max_q_s_prime_dyna = np.max(self.Q[s_prime_dyna], axis=1)
        self.Q[s_dyna, a_dyna] = (1 - self.alpha) * self.Q[s_dyna, a_dyna] + self.alpha * (
                    r_dyna + self.gamma * max_q_s_prime_dyna)


if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")

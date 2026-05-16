from level import Level
from math import sqrt
import settings as config
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim



def distance(px, py, rx, ry):
    return (sqrt((px - rx)**2 + (py-ry)**2))

def get_nearest_platforms(obs, amount=4):
    player_x = obs["player_center_x"]
    player_y = obs["player_center_y"]

    transformed = []

    for p in obs["platforms"]:
        dx = p["center_x"] - player_x
        dy = p["center_y"] - player_y

        transformed.append({
            **p,
            "dx": dx,
            "dy": dy,
            "distance": distance(player_x, player_y, p["center_x"], p["center_y"]),
        })

    # Prefer platforms above the player.
    above_player = [p for p in transformed if p["dy"] < 0]

    if above_player:
        transformed = above_player

    transformed.sort(key=lambda p: p["distance"])

    return transformed[:amount]

def get_observation(player):
    obs = {
        "player_x": player.rect.x,
        "player_y": player.rect.y,
        "player_center_x": player.rect.centerx,
        "player_center_y": player.rect.centery,
        "player_vx": player._velocity.x,
        "player_vy": player._velocity.y,
        "gravity": player.gravity,
        "accel": player.accel,
        "platforms": []
    }

    lvl = Level.instance
    if lvl:
        for p in lvl.platforms:
            obs["platforms"].append({
                "x": p.rect.x,
                "y": p.rect.y,
                "center_x": p.rect.centerx,
                "center_y": p.rect.centery,
                "w": p.rect.width,
                "h": p.rect.height,
                "breakable": int(p.breakable),
                "has_booster": int(p.bonus is not None),
                "booster_x": p.bonus.rect.x if p.bonus else None,
                "booster_y": p.bonus.rect.y if p.bonus else None,
                "booster_force": p.bonus.force if p.bonus else 0,
            })

    return obs

def obs_to_vector(obs, amount=4):
    nearest_platforms = get_nearest_platforms(obs, amount=amount)

    vector = [
        obs["player_center_x"] / config.XWIN,
        obs["player_center_y"] / config.YWIN,
        obs["player_vx"] / config.PLAYER_MAX_SPEED,
        obs["player_vy"] / 100,
    ]

    for p in nearest_platforms:
        dx = p["dx"] / config.XWIN
        dy = p["dy"] / config.YWIN

        vector.extend([
            dx,
            dy,
            p["breakable"],
            p["has_booster"],
        ])

    missing_platforms = amount - len(nearest_platforms)

    for _ in range(missing_platforms):
        vector.extend([
            0.0,  # dx
            0.0,  # dy
            0.0,  # breakable
            0.0,  # has_booster
        ])

    return vector




def heuristic(obs):
    nearest_platforms = get_nearest_platforms(obs, amount=4)

    if not nearest_platforms:
        return 0

    target = nearest_platforms[0]

    player_x = obs["player_center_x"]
    target_x = target["center_x"]

    dead_zone = 10

    if target_x < player_x - dead_zone:
        return 1  # left

    if target_x > player_x + dead_zone:
        return 2  # right

    return 0  # stay still



class DQN(nn.Module):
    def __init__(self, state_size=20, action_size=3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_size=20, action_size=3, frame_stack_size=4):
        self.state_size = state_size
        self.action_size = action_size
        self.frame_stack_size = frame_stack_size
        self.stacked_state_size = state_size * frame_stack_size

        self.gamma = 0.99
        self.lr = 0.001

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.batch_size = 64
        self.memory = deque(maxlen=50_000)

        self.model = DQN(self.stacked_state_size, action_size)
        self.target_model = DQN(self.stacked_state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        self.target_update_frequency = 100
        self.learn_step_counter = 0

    def save(self, path):
        torch.save({
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "memory": self.memory,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.target_model.load_state_dict(checkpoint["target_model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.memory = checkpoint["memory"]


    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(state_tensor)

        return torch.argmax(q_values).item()


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def _update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_frequency == 0:
            self._update_target_model()

        batch = random.sample(self.memory, self.batch_size)

        states = torch.tensor([x[0] for x in batch], dtype=torch.float32)
        actions = torch.tensor([x[1] for x in batch], dtype=torch.long)
        rewards = torch.tensor([x[2] for x in batch], dtype=torch.float32)
        next_states = torch.tensor([x[3] for x in batch], dtype=torch.float32)
        dones = torch.tensor([x[4] for x in batch], dtype=torch.bool)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (~dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

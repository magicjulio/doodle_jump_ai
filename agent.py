from camera import Camera
from level import Level
from math import sqrt
import settings as config
import random
import os
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim



PLATFORM_OBSERVATION_COUNT = 4
STATE_SIZE = 4 + PLATFORM_OBSERVATION_COUNT * 2
FEATURE_VERSION = 5


def wrap_delta(target_x, player_x, width=config.XWIN):
    dx = target_x - player_x
    half_width = width / 2

    if dx > half_width:
        dx -= width
    elif dx < -half_width:
        dx += width

    return dx


def distance(dx, dy):
    return sqrt(dx**2 + dy**2)

def get_nearest_platforms(obs, amount=PLATFORM_OBSERVATION_COUNT):
    player_x = obs["player_center_x"]
    player_y = obs["player_screen_center_y"]
    player_vy = obs["player_vy"]

    transformed = []

    for p in obs["platforms"]:
        dx = wrap_delta(p["center_x"], player_x)
        dy = p["screen_center_y"] - player_y

        transformed.append({
            **p,
            "dx": dx,
            "dy": dy,
            "distance": distance(dx, dy),
        })

    if player_vy > 0:
        # While falling, the next useful platform is usually below the player.
        preferred = [p for p in transformed if p["dy"] > -config.PLAYER_SIZE[1]]
    else:
        preferred = [p for p in transformed if p["dy"] < config.YWIN * 0.35]

    if preferred:
        transformed = preferred

    transformed.sort(key=lambda p: (abs(p["dy"]) * 1.4 + abs(p["dx"]), p["distance"]))

    return transformed[:amount]

def get_observation(player):
    camera = Camera.instance
    camera_y = camera.state.y if camera else 0
    player_screen_y = player.rect.y - camera_y

    obs = {
        "player_x": player.rect.x,
        "player_y": player.rect.y,
        "player_center_x": player.rect.centerx,
        "player_center_y": player.rect.centery,
        "player_screen_y": player_screen_y,
        "player_screen_center_y": player.rect.centery - camera_y,
        "player_vx": player._velocity.x,
        "player_vy": player._velocity.y,
        "gravity": player.gravity,
        "accel": player.accel,
        "camera_y": camera_y,
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
                "screen_y": p.rect.y - camera_y,
                "screen_center_y": p.rect.centery - camera_y,
                "w": p.rect.width,
                "h": p.rect.height,
            })

    return obs

def obs_to_vector(obs, amount=PLATFORM_OBSERVATION_COUNT):
    nearest_platforms = get_nearest_platforms(obs, amount=amount)

    vector = [
        obs["player_center_x"] / config.XWIN,
        obs["player_screen_center_y"] / config.YWIN,
        obs["player_vx"] / config.PLAYER_MAX_SPEED,
        obs["player_vy"] / config.PLAYER_MAX_SPEED,
    ]

    for p in nearest_platforms:
        dx = p["dx"] / config.XWIN
        dy = p["dy"] / config.YWIN

        vector.extend([
            dx,
            dy,
        ])

    missing_platforms = amount - len(nearest_platforms)

    for _ in range(missing_platforms):
        vector.extend([
            0.0,  # dx
            0.0,  # dy
        ])

    return vector






class DQN(nn.Module):
    def __init__(self, state_size=STATE_SIZE, action_size=3):
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
    def __init__(self, state_size=STATE_SIZE, action_size=3, frame_stack_size=16):
        self.state_size = state_size
        self.action_size = action_size
        self.frame_stack_size = frame_stack_size
        self.stacked_state_size = state_size * frame_stack_size
        use_cuda = torch.cuda.is_available() and os.environ.get("FORCE_CPU", "0") != "1"
        self.device = torch.device("cuda" if use_cuda else "cpu")
        print("Using device:", self.device)

        self.gamma = 0.99
        self.lr = 0.0005

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay_episodes = max(
            1,
            int(os.environ.get("EPSILON_DECAY_EPISODES", "50000")),
        )
        self.epsilon_decay = self.epsilon_min ** (1 / self.epsilon_decay_episodes)
        self.batch_size = 64
        self.memory = deque(maxlen=50_000)
        self.replay_warmup = 1_000

        self.model = DQN(self.stacked_state_size, action_size).to(self.device)
        self.target_model = DQN(self.stacked_state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.target_update_frequency = 500
        self.learn_step_counter = 0
        self.feature_version = FEATURE_VERSION

    def save(self, path):
        torch.save({
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "memory": self.memory,
            "learn_step_counter": self.learn_step_counter,
            "feature_version": self.feature_version,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        print("[+] torch load")
        if checkpoint.get("feature_version") == self.feature_version:
            print("[+] checkpoint compatible")
            self.model.load_state_dict(checkpoint["model"])
            self.target_model.load_state_dict(checkpoint.get("target_model", checkpoint["model"]))
            self.epsilon = checkpoint.get("epsilon", self.epsilon)
            self.learn_step_counter = checkpoint.get("learn_step_counter", 0)

            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                for state in self.optimizer.state.values():
                    for key, value in state.items():
                        if torch.is_tensor(value):
                            state[key] = value.to(self.device)
            self.memory = checkpoint.get("memory", self.memory)
            return True
        else:
            self.memory.clear()
            self.epsilon = 1.0
            self.learn_step_counter = 0
            print("Started fresh because checkpoint observation features are stale.")
            return False


    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.inference_mode():
            q_values = self.model(state_tensor)

        return torch.argmax(q_values).item()


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


    def _update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def learn(self):
        if len(self.memory) < max(self.batch_size, self.replay_warmup):
            return

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_frequency == 0:
            self._update_target_model()

        batch = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.bool, device=self.device)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1)
            next_q = self.target_model(next_states).gather(
                1,
                next_actions.unsqueeze(1),
            ).squeeze(1)
            target_q = rewards + self.gamma * next_q * (~dones).float()

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()

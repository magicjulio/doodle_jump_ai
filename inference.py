import argparse
import os


def parse_args():
	parser = argparse.ArgumentParser(description="Run a trained DQN checkpoint without learning.")
	parser.add_argument("--checkpoint", default="dqn_checkpoint.pth")
	parser.add_argument("--episodes", type=int, default=5)
	parser.add_argument("--headless", action="store_true")
	return parser.parse_args()


args = parse_args()

if args.headless:
	os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
from collections import deque

from camera import Camera
from env import DoodleJumpEnv
from level import Level
from player import Player
from agent import DQNAgent, get_observation, obs_to_vector
import settings as config


class InferenceGame:
	def __init__(self, checkpoint_path, headless=False):
		self.headless = headless
		self.action_repeat = int(os.environ.get("ACTION_REPEAT", "4"))
		self.window = pygame.display.set_mode(config.DISPLAY, config.FLAGS)
		self.clock = pygame.time.Clock()

		self.camera = Camera()
		self.lvl = Level()
		self.player = Player(
			config.HALF_XWIN - config.PLAYER_SIZE[0] / 2,
			config.HALF_YWIN + config.HALF_YWIN / 2,
			*config.PLAYER_SIZE,
			config.PLAYER_COLOR,
		)

		self.env = DoodleJumpEnv(self)
		self.agent = DQNAgent(frame_stack_size=16)
		if not self.agent.load(checkpoint_path):
			print("Warning: checkpoint was skipped because its observation features are stale.")
		self.agent.epsilon = 0.0
		self.stacked_frames = deque(maxlen=self.agent.frame_stack_size)

		self.score = 0
		self.score_txt = config.SMALL_FONT.render("0 m", 1, config.GRAY)
		self.score_pos = pygame.math.Vector2(10, 10)
		self.gameover_txt = config.LARGE_FONT.render("Game Over", 1, config.GRAY)
		self.gameover_rect = self.gameover_txt.get_rect(
			center=(config.HALF_XWIN, config.HALF_YWIN)
		)
		self.alive = True

	def reset(self):
		self.camera.reset()
		self.lvl.reset()
		self.player.reset()
		self.stacked_frames.clear()

	def close(self):
		self.alive = False

	def _event_loop(self):
		if self.headless:
			return

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				self.close()
			elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
				self.close()

	def _stack_observation(self, obs):
		state = obs_to_vector(obs)

		if not self.stacked_frames:
			for _ in range(self.agent.frame_stack_size):
				self.stacked_frames.append(state)
		else:
			self.stacked_frames.append(state)

		return [item for frame in self.stacked_frames for item in frame]

	def _get_stacked_state(self, obs=None):
		if not self.stacked_frames:
			if obs is None:
				obs = get_observation(self.player)
			return self._stack_observation(obs)

		return [item for frame in self.stacked_frames for item in frame]

	def _step(self, render=False):
		state = self._get_stacked_state()
		action = self.agent.choose_action(state)
		done = False
		info = {"score": self.score}
		final_obs = None

		for _ in range(self.action_repeat):
			next_obs, _, done, info = self.env.step(action)
			final_obs = next_obs
			# self._stack_observation(next_obs)
			self._update_score_text(info)

			if render:
				self._render_loop()
				self._event_loop()

			if done:
				break
		
		if final_obs is not None:
			self._stack_observation(final_obs)
		
		self._update_score_text(info)

		return done, info

	def _update_score_text(self, info):
		if self.player.dead:
			return

		self.score = info["score"]
		if not self.headless:
			self.score_txt = config.SMALL_FONT.render(
				str(self.score) + " m", 1, config.GRAY
			)

	def _render_loop(self):
		if self.headless:
			return

		self.window.fill(config.WHITE)
		self.lvl.draw(self.window)
		self.player.draw(self.window)

		if self.player.dead:
			self.window.blit(self.gameover_txt, self.gameover_rect)
		self.window.blit(self.score_txt, self.score_pos)

		pygame.display.update()
		self.clock.tick(config.FPS)

	def run_episodes(self, episode_count):
		for episode in range(episode_count):
			if not self.alive:
				break

			self.env.reset()
			done = False
			last_info = {"score": 0}

			while self.alive and not done:
				self._event_loop()
				done, last_info = self._step(render=not self.headless)

			print("inference episode:", episode + 1, "score:", last_info["score"])

		pygame.quit()


if __name__ == "__main__":
	game = InferenceGame(args.checkpoint, headless=args.headless)
	game.run_episodes(args.episodes)

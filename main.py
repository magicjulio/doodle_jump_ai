import os
import csv

HEADLESS = os.environ.get("HEADLESS", "0") == "1"
if HEADLESS:
	os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame, sys
from collections import deque

from singleton import Singleton
from camera import Camera
from player import Player
from level import Level
import settings as config

# ai agent
from env import DoodleJumpEnv

from agent import DQNAgent, get_observation, obs_to_vector



class Game(Singleton):
	"""
	A class to represent the game.

	used to manage game updates, draw calls and user input events.
	Can be access via Singleton: Game.instance .
	(Check Singleton design pattern for more info)
	 """

	# constructor called on new instance: Game()
	def __init__(self) -> None:

		# ============= Initialisation =============
		self.__alive = True
		# Window / Render
		self.window = pygame.display.set_mode(config.DISPLAY,config.FLAGS)
		self.clock = pygame.time.Clock()

		# Instances
		self.camera = Camera()
		self.lvl = Level()
		self.player = Player(
			config.HALF_XWIN - config.PLAYER_SIZE[0]/2,# X POS
			config.HALF_YWIN + config.HALF_YWIN/2,#      Y POS
			*config.PLAYER_SIZE,# SIZE
			config.PLAYER_COLOR#  COLOR
		)

		# ai agent env
		self.env = DoodleJumpEnv(self)
		self.agent = DQNAgent(frame_stack_size=16)
		self.human_control = False
		self.headless = HEADLESS
		self.train_episodes_per_checkpoint = int(
			os.environ.get("TRAIN_EPISODES_PER_CHECKPOINT", "1000" if self.headless else "100")
		)
		self.eval_episodes = int(os.environ.get("EVAL_EPISODES", "0" if self.headless else "3"))
		self.action_repeat = int(os.environ.get("ACTION_REPEAT", "4"))
		self.checkpoint_path = "dqn_checkpoint.pth"
		self.progress_log_path = os.environ.get("PROGRESS_LOG_PATH", "training_progress.csv")
		self.total_episode_count = 0
		self.train_episode_count = 0
		self.eval_episode_count = 0
		self._init_progress_log()

		if os.path.exists(self.checkpoint_path):
			if self.agent.load(self.checkpoint_path):
				print("Loaded checkpoint:", self.checkpoint_path)
			else:
				print("Skipped stale checkpoint:", self.checkpoint_path)


		# User Interface
		self.score = 0
		self.score_txt = config.SMALL_FONT.render("0 m",1,config.GRAY)
		self.score_pos = pygame.math.Vector2(10,10)

		self.gameover_txt = config.LARGE_FONT.render("Game Over",1,config.GRAY)
		self.gameover_rect = self.gameover_txt.get_rect(
			center=(config.HALF_XWIN,config.HALF_YWIN))

		# Frame stacking
		self.stacked_frames = deque(maxlen=self.agent.frame_stack_size)


	def close(self):
		self.__alive = False


	def reset(self):
		self.camera.reset()
		self.lvl.reset()
		self.player.reset()
		self.stacked_frames.clear()


	def _event_loop(self):
		# ---------- User Events ----------
		if self.headless:
			return

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				self.close()
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					self.close()
				if event.key == pygame.K_RETURN and self.player.dead:
					# update env instead  of just self.reset() 
					self.env.reset()

			if self.human_control:
				self.player.handle_event(event)


	def _stack_observation(self, obs):
		state = obs_to_vector(obs)

		if not self.stacked_frames:
			for _ in range(self.agent.frame_stack_size):
				self.stacked_frames.append(state)
		else:
			self.stacked_frames.append(state)

		return list(self.stacked_frames)


	def _get_stacked_state(self, obs=None):
		if not self.stacked_frames:
			if obs is None:
				obs = get_observation(self.player)
			return self._stack_observation(obs)

		return list(self.stacked_frames)


	def _agent_step(self, training=True, render=False):
		obs = get_observation(self.player)
		state = self._get_stacked_state(obs)
		flat_state = [item for sublist in state for item in sublist]

		action = self.agent.choose_action(flat_state)

		done = False
		total_reward = 0
		info = {"score": self.score}
		final_obs = None

		for _ in range(self.action_repeat):
			next_obs, reward, done, info = self.env.step(action)
			final_obs = next_obs
			total_reward += reward

			if render:
				self._update_score_text(info)
				self._render_loop()
				self._event_loop()

			if done:
				break

		if final_obs is not None:
			self._stack_observation(final_obs)
		next_state = list(self.stacked_frames)
		flat_next_state = [item for sublist in next_state for item in sublist]

		if training:
			self.agent.remember(flat_state, action, total_reward, flat_next_state, done)
			self.agent.learn()

		self._update_score_text(info)

		return done, total_reward, info


	def _update_score_text(self, info):
		if self.player.dead:
			return

		self.score = info["score"]
		if not self.headless:
			self.score_txt = config.SMALL_FONT.render(
				str(self.score) + " m", 1, config.GRAY)


	def _update_loop(self):
		# ----------- Update -----------
		done, reward, info = self._agent_step(training=True)

		if done:
			self.env.reset()


	def _render_loop(self):
		# ----------- Display -----------
		self.window.fill(config.WHITE)
		self.lvl.draw(self.window)
		self.player.draw(self.window)

		# User Interface
		if self.player.dead:
			self.window.blit(self.gameover_txt,self.gameover_rect)# gameover txt
		self.window.blit(self.score_txt, self.score_pos)# score txt

		pygame.display.update()# window update
		self.clock.tick(config.FPS)# max loop/s


	def run(self):
		# ============= MAIN GAME LOOP =============
		state = self._get_stacked_state()
		flat_state = [item for sublist in state for item in sublist]


		print("state length: ", len(flat_state))
		while self.__alive:
			self.run_episodes(self.train_episodes_per_checkpoint, training=True, render=False)
			self.agent.save(self.checkpoint_path)
			print("Saved checkpoint:", self.checkpoint_path)
			if self.eval_episodes:
				self.run_visual_evaluation(self.eval_episodes)
		pygame.quit()


	def run_episodes(self, episode_count, training=True, render=False):
		for episode in range(episode_count):
			if not self.__alive:
				return

			self.env.reset()
			done = False
			episode_reward = 0
			last_info = {"score": 0}

			while self.__alive and not done:
				if render or not self.headless:
					self._event_loop()
				done, reward, last_info = self._agent_step(training=training, render=render)
				episode_reward += reward

			mode = "train" if training else "eval"
			mode_episode = self._next_mode_episode(training)
			self.total_episode_count += 1
			self._log_progress(
				mode=mode,
				mode_episode=mode_episode,
				score=last_info["score"],
				reward=episode_reward,
			)

			print(
				mode,
				"episode:",
				mode_episode,
				"score:",
				last_info["score"],
				"reward:",
				round(episode_reward, 2),
				"epsilon:",
				round(self.agent.epsilon, 4),
			)

			if training:
				self.agent.decay_epsilon()


	def _next_mode_episode(self, training):
		if training:
			self.train_episode_count += 1
			return self.train_episode_count

		self.eval_episode_count += 1
		return self.eval_episode_count


	def _init_progress_log(self):
		if not self.progress_log_path:
			return

		log_dir = os.path.dirname(self.progress_log_path)
		if log_dir:
			os.makedirs(log_dir, exist_ok=True)

		if os.path.exists(self.progress_log_path) and os.path.getsize(self.progress_log_path) > 0:
			self._load_progress_log_counts()
			return

		with open(self.progress_log_path, "w", newline="") as file:
			writer = csv.writer(file)
			writer.writerow([
				"episode",
				"mode",
				"mode_episode",
				"score",
				"reward",
				"epsilon",
			])


	def _load_progress_log_counts(self):
		try:
			with open(self.progress_log_path, "r", newline="") as file:
				reader = csv.DictReader(file)
				for row in reader:
					self.total_episode_count = max(
						self.total_episode_count,
						int(row.get("episode") or 0),
					)
					mode_episode = int(row.get("mode_episode") or 0)
					if row.get("mode") == "train":
						self.train_episode_count = max(self.train_episode_count, mode_episode)
					elif row.get("mode") == "eval":
						self.eval_episode_count = max(self.eval_episode_count, mode_episode)
		except (OSError, ValueError):
			print("Could not restore progress counters from:", self.progress_log_path)


	def _log_progress(self, mode, mode_episode, score, reward):
		if not self.progress_log_path:
			return

		with open(self.progress_log_path, "a", newline="") as file:
			writer = csv.writer(file)
			writer.writerow([
				self.total_episode_count,
				mode,
				mode_episode,
				score,
				round(reward, 6),
				round(self.agent.epsilon, 6),
			])


	def run_visual_evaluation(self, episode_count):
		old_epsilon = self.agent.epsilon
		self.agent.epsilon = 0.0
		self.run_episodes(episode_count, training=False, render=True)
		self.agent.epsilon = old_epsilon



if __name__ == "__main__":

	game = Game()
	game.run()

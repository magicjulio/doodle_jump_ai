import os

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
		self.agent = DQNAgent()
		self.human_control = False
		self.headless = HEADLESS
		self.train_episodes_per_checkpoint = int(
			os.environ.get("TRAIN_EPISODES_PER_CHECKPOINT", "1000" if self.headless else "100")
		)
		self.eval_episodes = int(os.environ.get("EVAL_EPISODES", "0" if self.headless else "3"))
		self.checkpoint_path = "dqn_checkpoint.pth"

		if os.path.exists(self.checkpoint_path):
			self.agent.load(self.checkpoint_path)
			print("Loaded checkpoint:", self.checkpoint_path)


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


	def _get_stacked_state(self):
		obs = get_observation(self.player)
		state = obs_to_vector(obs)

		if not self.stacked_frames:
			for _ in range(self.agent.frame_stack_size):
				self.stacked_frames.append(state)
		else:
			self.stacked_frames.append(state)

		return list(self.stacked_frames)


	def _agent_step(self, training=True):
		state = self._get_stacked_state()
		flat_state = [item for sublist in state for item in sublist]

		action = self.agent.choose_action(flat_state)

		next_obs, reward, done, info =  self.env.step(action)
		
		next_state = self._get_stacked_state()
		flat_next_state = [item for sublist in next_state for item in sublist]

		if training:
			self.agent.remember(flat_state, action, reward, flat_next_state, done)
			self.agent.learn()

		if not self.player.dead:
			self.score = info["score"]
			if not self.headless:
				self.score_txt = config.SMALL_FONT.render(
					str(self.score) + " m", 1, config.GRAY)

		return done, reward, info


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


		print("state length: ", len(flat_state), "state:", flat_state)
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
				done, reward, last_info = self._agent_step(training=training)
				episode_reward += reward

				if render:
					self._render_loop()

			mode = "train" if training else "eval"
			print(
				mode,
				"episode:",
				episode + 1,
				"score:",
				last_info["score"],
				"reward:",
				round(episode_reward, 2),
				"epsilon:",
				round(self.agent.epsilon, 4),
			)


	def run_visual_evaluation(self, episode_count):
		old_epsilon = self.agent.epsilon
		self.agent.epsilon = 0.0
		self.run_episodes(episode_count, training=False, render=True)
		self.agent.epsilon = old_epsilon




if __name__ == "__main__":
	# ============= PROGRAM STARTS HERE =============
	game = Game()
	game.run()

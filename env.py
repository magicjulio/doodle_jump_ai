from agent import get_observation


class DoodleJumpEnv:
    def __init__(self, game):
        self.game = game
        self.previous_score = 0

    def reset(self):
        self.game.reset()

        obs = get_observation(self.game.player)
        self.previous_score = self.get_score()

        return obs

    def get_score(self):
        return -self.game.camera.state.y // 50

    def get_reward(self, current_score=None):
        if current_score is None:
            current_score = self.get_score()

        reward = current_score - self.previous_score
        done = self.game.player.dead
        if done:
            reward -= 100

        self.previous_score = current_score

        return reward, done

    def step(self, action):
        self.game.player.set_action(action)

        self.game.player.update()
        self.game.lvl.update()

        if not self.game.player.dead:
            self.game.camera.update(self.game.player.rect)

        obs = get_observation(self.game.player)
        current_score = self.get_score()
        reward, done = self.get_reward(current_score)

        info = {
            "score": current_score
        }

        return obs, reward, done, info

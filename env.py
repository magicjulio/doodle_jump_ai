from agent import get_observation


class DoodleJumpEnv:
    def __init__(self, game):
        self.game = game
        self.previous_score = 0
        self.best_score = 0
        self.frames_without_progress = 0
        self.max_stuck_frames = 300
        self.visited_platforms = set()



    def reset(self):
        self.game.reset()
        
        self.previous_score = self.get_score()
        self.best_score = self.previous_score
        self.frames_without_progress = 0
        self.visited_platforms.clear()

        return get_observation(self.game.player)

    def get_score(self):
        return -self.game.camera.state.y // 50

    def get_reward(self):
        current_score = self.get_score()

        if current_score > self.best_score:
            self.best_score = current_score
            self.frames_without_progress = 0
        else:
            self.frames_without_progress += 1

        stuck = self.frames_without_progress >= self.max_stuck_frames
        done = self.game.player.dead or stuck

        
        
        
        
        reward = current_score - self.previous_score

        landed_platform = self.game.player.last_landed_platform
        if landed_platform and landed_platform not in self.visited_platforms:
            reward += 10
            self.visited_platforms.add(landed_platform)

        if self.game.player.dead:
            reward -= 100

        if stuck:
            reward -= 120

        self.previous_score = current_score

        return reward, done

    def step(self, action):
        self.game.player.set_action(action)

        self.game.player.update()
        self.game.lvl.update()

        if not self.game.player.dead:
            self.game.camera.update(self.game.player.rect)


        

        obs = get_observation(self.game.player)
        reward, done = self.get_reward()
        

        info = {
            "score": self.get_score()
        }

        return obs, reward, done, info

import numpy as np
import gym, os, math, random, pygame, sys, time
import matplotlib.pyplot as plt
import pygame.freetype
from gym import spaces
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ppo1 import PPO1
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import PPO2, DQN

# ffmpeg -r 20 -i frame_%01d.jpeg -c:v libx264 -vf fps=20 -pix_fmt yuv420p out.mp4

class Projectile():

    tps = 0.4
    life = 35
    is_alive = True

    def __init__(self, shootoutenv, ownerid, id, position, rotation):
        self.shootoutenv = shootoutenv
        self.ownerid = ownerid
        self.id = id
        self.position = position
        self.rotation = rotation

    def step(self):
        #print(f"bullet w/ {self.ownerid}, {self.position}")
        self.life -= 1
        if self.life <= 0 or math.sqrt(self.position[0]**2 + self.position[1]**2) >= self.shootoutenv.arena_radius:
            self.is_alive = False
        else:
            self.position[0] += math.cos(self.rotation) * self.tps
            self.position[1] += math.sin(self.rotation) * self.tps

class Player():

    hitbox_radius = 0.5
    stock = 3

    states = ("neutral", "dash", "dodge", "reload", "fire", "carrying", "spawning")

    def __init__(self, shootoutenv, id, health=1, max_ammo=10, tps=0.1, position=[0,0], rotation=0):
        self.shootoutenv = shootoutenv
        self.id = id
        self.health = health
        self.start_health = health
        self.max_ammo = max_ammo
        self.ammo = max_ammo
        self.tps = tps
        self.position = position
        self.start_pos = position[:]
        self.rotation = rotation # radians, 0 deg facing right
        self.start_rotation = rotation
        self.state = ["neutral", 0] # neutral dash dodge reload fire carrying spawning
        self.cooldown = 0
        self.dodgetimer = 0

    def get_rect(self):
        return pygame.Rect()
    def reset(self):
        # return player to start pos
        self.position = self.start_pos[:]
        self.rotation = self.start_rotation
        self.state = ["neutral", 0] if self.stock == 3 else ["carrying", 0]
        self.ammo = self.max_ammo
        self.cooldown = 0
        self.dodgetimer = 0

    def step(self, action):
        # update state
        for i in range(len(action)):
            if i != 4:
                action[i] = round(action[i])

        #print(f"{self.id}, {self.position}, {self.rotation}, {self.cooldown}, {self.state}")
        if self.state[0] in ["neutral", "spawning"] and self.cooldown == 0:
            # able to change state
            if action[7]:
                if self.ammo <= 0:
                    self.state = ["reload", 0]
                else:
                    self.state = ["fire", 0]
                    # id will go from 10-1 inclusive
                    to_pop = []
                    for i in range(len(self.shootoutenv.projectiles)):
                        projectile = self.shootoutenv.projectiles[i]
                        if projectile.ownerid == self.id and projectile.id == self.ammo:
                            to_pop.append(i) # remove duplicate bullet
                    for i in to_pop:
                        self.shootoutenv.projectiles.pop(i)
                    self.shootoutenv.projectiles.append(Projectile(self.shootoutenv, self.id, self.ammo, self.position[:], self.rotation))
                    self.ammo -= 1
            elif action[5]:
                # dash
                self.state = ["dash", 0]
            elif action[6] and self.dodgetimer == 0:
                self.state = ["dodge", 0]
                self.dodgetimer = 24



        # check state
        if self.state[0] in ["neutral", "fire", "reload", "dash", "spawning"]:
            # A D
            # 1 0 -1
            # 0 0 0
            # 0 1 1
            # 1 1 1
            var_tps = self.tps
            if self.state[0] == "dash":
                horizontal = 0
                vertical = 1
                projection_of_normalized = 1
                var_tps *= 3
                self.state[1] += 1
                if self.state[1] >= 5: # dash frames
                    self.state = ["neutral", 0]
                    self.cooldown = 3
            elif self.state[0] != "dash":
                if self.cooldown == 0:
                    horizontal = 1 if action[3] else (-1 if action[1] else 0)

                    vertical = 1 if action[0] else (-1 if action[2] else 0)
                    projection_of_normalized = 1 if abs(horizontal) + abs(vertical)==1 else math.sqrt(2)

                    self.rotation = action[4]*2*math.pi
                else:
                    horizontal = 0
                    vertical = 0
                    projection_of_normalized = 1

                    self.rotation = action[4]*2*math.pi

            if self.state[0] == "reload":
                var_tps *= 0.6 # speed debuff
                self.state[1] += 1
                if self.state[1] >= 12: # dash frames
                    self.state = ["neutral", 0]
                    self.ammo = self.max_ammo




            self.position[0] += (math.cos(self.rotation) * vertical + math.sin(self.rotation) * horizontal) * projection_of_normalized * var_tps
            self.position[1] += (math.sin(self.rotation) * vertical - math.cos(self.rotation) * horizontal) * projection_of_normalized * var_tps

        # progress state
        if self.state[0] == "dodge":
            self.state[1] += 1
            if self.state[1] >= 8: # dodge frames
                self.state = ["neutral", 0]
                self.cooldown = 3
        elif self.state[0] == "fire":
            self.state[1] += 1
            if self.state[1] >= 8: # fire frames
                self.state = ["neutral", 0]
        elif self.state[0] == "carrying":
            self.state[1] += 1
            if self.state[1] >= 15:
                self.state = ["spawning", 0]
        elif self.state[0] == "spawning":
            self.state[1] += 1
            if self.state[1] >= 15:
                self.state = ["neutral", 0]

        self.cooldown = max(0, self.cooldown - 1) # lower cooldown
        self.dodgetimer = max(0, self.dodgetimer - 1)

        # check if dead
        return math.sqrt(self.position[0]**2 + self.position[1]**2) >= self.shootoutenv.arena_radius

class BaselinePolicy:
    def predict(self,obs):
        #print("Sampling random action.")
        #out = np.array([1,0,0,0,0,1,0,0])
        out = spaces.Box(low=np.array([0,0,0,0,0,0,0,0]), high=np.array([1,1,1,1,1,1,1,1]), shape=(8,), dtype=np.float32).sample()
        for i in range(len(out)):
            if i != 4:
                out[i] = round(out[i])
        #print(obs)
        return out

class PlayerPolicy:

    def __init__(self, pixels_per_tile):
        self.pixels_per_tile = pixels_per_tile

    def game_to_pixel_coords(self, coords):
        return (200 + coords[0] * self.pixels_per_tile, 200 - coords[1] * self.pixels_per_tile)

    def predict(self, obs):
        # W A S D rotation dash dodge fire
        out = np.array([0,0,0,0,0,0,0,0], dtype=np.float32)
        keys=pygame.key.get_pressed()
        if keys[pygame.K_w]:
            out[0] = 1
        if keys[pygame.K_a]:
            out[1] = 1
        if keys[pygame.K_s]:
            out[2] = 1
        if keys[pygame.K_d]:
            out[3] = 1
        if keys[pygame.K_SPACE] or keys[pygame.K_LSHIFT]:
            out[5] = 1
        mpos = pygame.mouse.get_pos()
        ppos = (obs[1], obs[2])
        ppos = self.game_to_pixel_coords(ppos)
        #print(ppos)
        #print(f"y:{mpos[1] - ppos[1]} / x:{mpos[0] - ppos[0]}")
        out[4] = np.float32((math.atan2(ppos[1] - mpos[1], mpos[0] - ppos[0])/(2*math.pi))%1)
        if pygame.mouse.get_pressed()[2]:
            out[6] = 1
        if pygame.mouse.get_pressed()[0]:
            out[7] = 1
        #print(out)
        #print(obs)
        return out

class ShootoutEnv(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, time_limit=1000):
        super(ShootoutEnv, self).__init__()

        self.arena_radius = 5
        self.time_limit = time_limit
        self.projectiles = []
        self.timer = 0

        pygame.init()
        self.GAME_FONT = pygame.freetype.Font("media/Consolas.ttf", 16)

        self.screen_width = 400

        self.size = (self.screen_width, self.screen_width)
        self.screen_width_tiles = 2 * self.arena_radius / 0.75
        self.pixels_per_tile = int((self.screen_width*0.375)/(self.arena_radius))

        self.screen = pygame.display.set_mode(self.size)


        self.player_sprite = pygame.image.load("media/baseball.png")
        self.player_sprite = pygame.transform.scale(self.player_sprite, (self.pixels_per_tile, self.pixels_per_tile))
        self.max_ammo = 10

        # W A S D rotation dash dodge fire
        self.action_space = spaces.Box(low=np.array([0,0,0,0,0,0,0,0]), high=np.array([1,1,1,1,1,1,1,1]), shape=(8,), dtype=np.float32)

        # self(health absposx abspoxy centerdist absrot ammo stateval stock
        # state
        # rotationtoenemy
        # near bullets) [each bullet: exists relativeposx relativeposy rotation facing_rot]
        # other(health absposx absposy centerdist relativeposx relativeposy orienposx orienposy
        # absrot relativerot stateval stock
        # state
        # near bullets) [each bullet: exists relativeposx relativeposy rotation facing_rot]
        # timer
        lowarray = np.array(
        [0, -self.screen_width_tiles/2, -self.screen_width_tiles/2, 0, 0, 0, 0, 0] +
        [0 for _ in range(len(Player.states))] +
        [0] +
        [(0, -self.screen_width_tiles, -self.screen_width_tiles, 0, 0)[i%5] for i in range(self.max_ammo*5)] +
        [0, -self.screen_width_tiles/2, -self.screen_width_tiles/2, 0, -self.screen_width_tiles, -self.screen_width_tiles, -self.screen_width_tiles, -self.screen_width_tiles,
        0, 0, 0, 0] +
        [0 for _ in range(len(Player.states))] +
        [(0, -self.screen_width_tiles, -self.screen_width_tiles, 0, 0)[i%5] for i in range(self.max_ammo*5)] +
        [0]
        )
        higharray = np.array(
        [1, self.screen_width_tiles/2, self.screen_width_tiles/2, self.screen_width_tiles/2, 2 * math.pi, 10, 20, 3] +
        [1 for _ in range(len(Player.states))] +
        [2*math.pi] +
        [(1, self.screen_width_tiles, self.screen_width_tiles, 2*math.pi, 2*math.pi)[i%5] for i in range(self.max_ammo*5)] +
        [1, self.screen_width_tiles/2, self.screen_width_tiles/2, self.screen_width_tiles/2, self.screen_width_tiles, self.screen_width_tiles, self.screen_width_tiles, self.screen_width_tiles,
        2 * math.pi, 2 * math.pi, 20, 3] +
        [1 for _ in range(len(Player.states))] +
        [(1, self.screen_width_tiles, self.screen_width_tiles, 2*math.pi, 2*math.pi)[i%5] for i in range(self.max_ammo*5)] +
        [self.time_limit]
        )
        print(f"shape of lowarray {lowarray.shape}")
        self.observation_space = spaces.Box(
        low=lowarray,
        high=higharray,
        shape=lowarray.shape,
        dtype=np.float32)

        self.otherAction = None

        #self.policy = BaselinePolicy()
        self.policy = PlayerPolicy(self.pixels_per_tile)


        pass


    def generate_observation(self, id):
        # self(health absposx abspoxy centerdist absrot ammo stateval stock
        # state
        # rotationtoenemy
        # near bullets) [each bullet: exists relativeposx relativeposy rotation facing_rot]
        # other(health absposx absposy centerdist relativeposx relativeposy orienposx orienposy
        # absrot relativerot stateval stock
        # state
        # near bullets) [each bullet: exists relativeposx relativeposy rotation facing_rot]
        # timer
        player = self.player_1 if self.player_1.id == id else self.player_2
        other_player = self.player_2 if self.player_1.id == id else self.player_1

        my_projectiles = [0 for _ in range(self.max_ammo)]
        other_projectiles = [0 for _ in range(self.max_ammo)]

        for projectile in self.projectiles:
            (my_projectiles if projectile.ownerid == id else other_projectiles)[projectile.id-1] = projectile

        obs = [player.health, player.position[0], player.position[1], math.sqrt(player.position[0]**2 + player.position[1]**2), player.rotation, player.ammo, player.state[1], player.stock]
        obs += [1 if player.state[0] == state else 0 for state in player.states]
        obs += [math.atan2(player.position[1] - other_player.position[1], other_player.position[0] - player.position[0])%(2*math.pi)]
        # bullet stuff mine
        for i in range(self.max_ammo):
            p = my_projectiles[i]
            result = [0,0,0,0,0]
            if p != 0:
                result = [1, p.position[0] - player.position[0], p.position[1] - player.position[1], p.rotation, (math.atan2(player.position[1] - p.position[1], p.position[0] - player.position[0])+p.rotation)%(2*math.pi)]
            obs += result

        offset = (other_player.position[0] - player.position[0], other_player.position[1] - player.position[1])
        e1 = (math.sin(player.rotation), -math.cos(player.rotation))
        e2 = (math.cos(player.rotation), math.sin(player.rotation))
        orienposx = offset[0]*e1[0] + offset[1]*e1[1]
        orienposy = offset[0]*e2[0] + offset[1]*e2[1]
        obs += [other_player.health, other_player.position[0], other_player.position[1], math.sqrt(other_player.position[0]**2 + other_player.position[1]**2), offset[0], offset[1], orienposx, orienposy]
        obs += [other_player.rotation, (other_player.rotation - player.rotation)%(2*math.pi), other_player.state[1], player.stock]
        obs += [1 if other_player.state[0] == state else 0 for state in other_player.states]
        # bullet stuff other
        for i in range(self.max_ammo):
            p = other_projectiles[i]
            result = [0,0,0,0,0]
            if p != 0:
                result = [1, p.position[0] - player.position[0], p.position[1] - player.position[1], p.rotation, (math.atan2(player.position[1] - p.position[1], p.position[0] - player.position[0])+p.rotation)%(2*math.pi)]
            obs += result
        obs += [self.timer]

        obs = np.array(obs)
        #print(f"shape of obs {obs.shape}")
        #print(obs)

        return obs

    override_flipper = 0

    def reset(self):
        self.timer = 0

        sideflipper = random.randint(0,1)*2 - 1
        sideflipper = self.override_flipper if self.override_flipper != 0 else sideflipper
        self.player_1 = Player(self, 1, 1, self.max_ammo, 0.1, [0, -sideflipper * self.arena_radius/2], sideflipper * math.pi * 0.5)
        self.player_2 = Player(self, 2, 1, self.max_ammo, 0.1, [0,sideflipper * self.arena_radius/2], -sideflipper * math.pi*0.5)

        self.player_1.stock = 3
        self.player_1.reset()

        self.player_2.stock = 3
        self.player_2.reset()

        self.projectiles = []



        return self.generate_observation(1)

    def game_to_pixel_coords(self, coords):
        return (200 + coords[0] * self.pixels_per_tile, 200 - coords[1] * self.pixels_per_tile)

    def center_coords(self, coords, width, height):
        return (coords[0]-width/2, coords[1] - height/2)

    def ko_player(self, player):
        player.stock -= 1
        if player.stock <= 0:
            return True
        else:
            # return player to start pos
            player.reset()
            self.projectiles = []
            return False

    def step(self, action):
        info = {}
        reward = 0
        done = False

        # step both players
        kod_players = [] # for skipping bullets later

        obs2 = self.generate_observation(2)
        prediction = self.policy.predict(obs2)
        if len(prediction) == 2 and len(prediction[0]) > 2:
            prediction = prediction[0]
        player_2_kod = self.player_2.step(prediction)

        player_1_kod = self.player_1.step(action)


        if player_1_kod and not player_2_kod:
            kod_players.append(self.player_1)
            if self.ko_player(self.player_1):
                return self.generate_observation(1), -3 + self.timer/self.time_limit, True, info
            else:
                reward -= 1

        elif player_2_kod and not player_1_kod:
            kod_players.append(self.player_2)
            if self.ko_player(self.player_2):
                return self.generate_observation(1), 3 - self.timer/self.time_limit, True, info
            else:
                reward += 1
        elif player_1_kod and player_2_kod:
            kod_players.append(self.player_1)
            kod_players.append(self.player_2)
            p1_lost = self.ko_player(self.player_1)
            p2_lost = self.ko_player(self.player_2)
            if p1_lost and p2_lost:
                return self.generate_observation(1), 0, True, info
            else:
                if p1_lost:
                    return self.generate_observation(1), -3 + self.timer/self.time_limit, True, info
                else:
                    reward -= 1
                if p2_lost:
                    return self.generate_observation(1), 3 - self.timer/self.time_limit, True, info
                else:
                    reward += 1


        # check if ko'd by projectile
        to_pop = []
        for projectile in self.projectiles:
            # step projectiles
            projectile.step()
            if not projectile.is_alive:
                to_pop.append(projectile)
                continue

            # if still alive, check hits
            killable_player = self.player_1 if projectile.ownerid == 2 else self.player_2
            if killable_player not in kod_players and killable_player.state[0] not in ["dodge", "carrying", "spawning"]:
                if math.sqrt((killable_player.position[0]-projectile.position[0])**2 + \
                (killable_player.position[1] - projectile.position[1])**2) <= killable_player.hitbox_radius:
                    if self.ko_player(killable_player):
                        return self.generate_observation(1), (3 - self.timer/self.time_limit) if projectile.ownerid == 1 else (-3 + self.timer/self.time_limit), True, info
                    else:
                        reward += 1 if projectile.ownerid == 1 else -1

                    # remove projectile
                    to_pop.append(projectile)
        # remove projectiles to remove
        for projectile in to_pop:
            if projectile in self.projectiles:
                self.projectiles.remove(projectile)

        # update obs if not killed



        self.timer += 1
        if self.timer >= self.time_limit:
            return self.generate_observation(1), reward, True, info
        else:
            return self.generate_observation(1), reward, done, info

    def color_from_state(self, state, cooldown):
        if state[0] == "neutral":
            if cooldown == 0:
                return (255,255,255)
            else:
                return (0, 255, 13) if self.timer % 2 else (50, 168, 56)
        elif state[0] == "reload":
            return (252, 148, 3) if state[1] % 2 else (222, 118, 0)
        elif state[0] == "fire":
            return (255, 42, 0) if state[1] % 2 else (222, 37, 0)
        elif state[0] == "dodge":
            return (191, 0, 255) if state[1] % 2 else (132, 0, 176)
        elif state[0] == "dash":
            return (0, 255, 251) if state[1] % 2 else (0, 204, 255)
        elif state[0] == "carrying":
            return (173, 16, 105)
        elif state[0] == "spawning":
            return (251, 255, 0)
        else:
            return (255,255,255)
    def render(self, mode='light'):
        if mode not in ['light', 'heavy']:
            raise NotImplementedError()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()

        self.screen.fill((0,0,0))

        pygame.draw.circle(self.screen, (255, 255, 255), (self.screen_width/2,self.screen_width/2), self.arena_radius * self.pixels_per_tile, width=1)

        self.GAME_FONT.render_to(self.screen, (0, 0), f"P1: ({self.player_1.stock}, {self.player_1.ammo}), P2: ({self.player_2.stock}, {self.player_2.ammo})", (255, 255, 255))
        self.GAME_FONT.render_to(self.screen, (0, self.screen_width-16), f"{self.timer}/{self.time_limit}", (255, 255, 255))

        player_rect = self.player_sprite.get_rect()

        self.screen.blit(self.player_sprite, player_rect.move(self.center_coords(self.game_to_pixel_coords(self.player_1.position),self.pixels_per_tile,self.pixels_per_tile)))

        pygame.draw.circle(self.screen, self.color_from_state(self.player_1.state, self.player_1.cooldown), self.game_to_pixel_coords(self.player_1.position), self.pixels_per_tile * 0.8, width=2)

        pygame.draw.circle(self.screen, (0,0,255), self.game_to_pixel_coords(self.player_1.position), 5)

        self.screen.blit(self.player_sprite, player_rect.move(self.center_coords(self.game_to_pixel_coords(self.player_2.position),self.pixels_per_tile,self.pixels_per_tile)))

        pygame.draw.circle(self.screen, self.color_from_state(self.player_2.state, self.player_2.cooldown), self.game_to_pixel_coords(self.player_2.position), self.pixels_per_tile * 0.8, width=2)

        pygame.draw.circle(self.screen, (255,0,0), self.game_to_pixel_coords(self.player_2.position), 5)

        for projectile in self.projectiles:
            pygame.draw.circle(self.screen, (255,0,0), self.game_to_pixel_coords(projectile.position), 1)

        if mode == 'heavy':
            if self.timer == 0:
                # first frame...
                self.GAME_FONT.render_to(self.screen, self.game_to_pixel_coords(self.player_1.position), f"Player 1", (0, 0, 255))
                self.GAME_FONT.render_to(self.screen, self.game_to_pixel_coords(self.player_2.position), f"Player 2", (255, 0, 0))

            if self.player_1.stock <= 0:
                self.GAME_FONT.render_to(self.screen, (self.screen_width/2, self.screen_width/2), f"Player 2 wins!", (255, 0, 0))

            elif self.player_2.stock <= 0:
                self.GAME_FONT.render_to(self.screen, (self.screen_width/2, self.screen_width/2), f"Player 1 wins!", (0, 0, 255))

        pygame.display.flip()
        if mode == 'heavy':
            if self.timer == 0:
                time.sleep(2)
            if self.player_1.stock <= 0:
                time.sleep(2)
            elif self.player_2.stock <= 0:
                time.sleep(2)

        pygame.image.save(self.screen, f"render/frame_{self.timer:03}.jpeg")

    def close(self):
        pass

LOGDIR = "models/ppo1_selfplay"

def test_player_equivalency():
    trial_count = 200
    policy1, policy2 = None, None
    env = ShootoutEnv()

    # load model if it's there
    modellist = [f for f in os.listdir(LOGDIR) if f.startswith("history")]
    modellist.sort()
    filename = None
    if len(modellist) > 0:
        filename = os.path.join(LOGDIR, modellist[-1]) # the latest best model
        if filename != None:
            print("loading model: ", filename)
            best_model_filename = filename
            policy1 = PPO1.load(filename, env=env)
            #policy2 = PPO1.load(filename, env=env)

    policy2 = BaselinePolicy()

    env.policy = policy1
    policy = policy2

    done = False
    total_reward = 0
    obs = env.reset()
    counter = trial_count

    #env.override_flipper = 1
    round_reward = 0
    while counter > 0:
        #action, _states = policy.predict(obs)
        action = policy.predict(obs)
        obs, reward, done, _ = env.step(action)
        env.render()
        time.sleep(0.05)

        total_reward += reward
        round_reward += reward
        if done:
            counter -= 1
            #ax.plot(trial_count-counter,round_reward)
            #print(round_reward)
            round_reward = 0
            obs = env.reset()
            #env.override_flipper = 1

    print(f"{total_reward}, {total_reward/trial_count}")

    done = False
    total_reward = 0
    counter = trial_count

    env.policy, policy = policy, env.policy
    env.reset()
    #env.override_flipper = -1

    while counter > 0:
        #action = policy.predict(obs)
        action, _states = policy.predict(obs)
        obs, reward, done, _ = env.step(action)
        #env.render()
        #time.sleep(0.05)

        total_reward += reward
        if done:
            counter -= 1
            obs = env.reset()
            #env.override_flipper = -1
    print(f"{total_reward}, {total_reward/trial_count}")

def player_vs_best_model():
    env = ShootoutEnv()
    obs = env.reset()
    env.render()

    # load model if it's there
    modellist = [f for f in os.listdir(LOGDIR) if f.startswith("history")]
    modellist.sort()
    if len(modellist) > 0:
        filename = os.path.join(LOGDIR, modellist[-1]) # the latest best model
        if filename != None:
            print("loading model: ", filename)
            best_model_filename = filename
            policy = PPO1.load(filename, env=env)
            #policy = BaselinePolicy()
            #pixels_per_tile = int((400*0.375)/(5))
            #policy = PlayerPolicy(pixels_per_tile)

    done = False
    total_reward = 0
    counter = 100

    while counter > 0:

        action, _states = policy.predict(obs)
        obs, reward, done, _ = env.step(action)

        total_reward += reward
        env.render()
        time.sleep(0.05)
        if done:
            counter -= 1
            obs = env.reset()
    print(total_reward)

def unused():
    env = ShootoutEnv()
    env.reset()
    env.render()
    for i in range(1000):
        obs, rewards, dones, info = env.step(BaselinePolicy().predict(None))
        env.render()
        time.sleep(0.05)
        if dones == True:
            print("Env complete! - - - - - - - - - - - -", "reward=", rewards)
            obs = env.reset()

if __name__ == "__main__":
    #player_vs_best_model()
    test_player_equivalency()

if __name__ == "__2main__":
    save_dir = "C:/Users/whmra/OneDrive/Documents/Python Projcs/STABLEBASELINES/1v1/models/"
    if True:

        env = ShootoutEnv()

        # Model stuff
        #kwargs = {'double_q': True, 'prioritized_replay': True, 'policy_kwargs': dict(dueling=True)}
        #model = DQN("MlpPolicy", env, verbose=1, **kwargs)
        model = PPO2(MlpPolicy, env, verbose=1)


        model.learn(total_timesteps=1000000) # 4000000
    else:
        #model = DQN.load(save_dir + "/DQN_5_5", verbose=1)
        print("Loading model from", save_dir)
        model = PPO2.load(save_dir + "PPO2_4_4(0)", verbose=1)

    if False:
        print("Training loaded model...")
        model.set_env(DummyVecEnv([lambda: ShootoutEnv()]))
        model.learn(total_timesteps=4000000)
    if True:
        # Create save dir
        print("saving model at", save_dir)
        os.makedirs(save_dir, exist_ok=True)
        model.save(save_dir + "PPO2_4_4(0)")
    # Show Model
    env = ShootoutEnv()
    obs = env.reset()
    wins = 0
    halts = 0
    losses = 0
    steps = 20000 # 20000
    display_steps = 100
    same_action_counter = 0
    last_action = 0
    # Demonstrate model
    print('Running model...')
    for i in range(steps):
        if i > steps-display_steps:
            # Display Maze
            print('')
            env.render()
        # Choose action
        #print(obs.shape)
        #print(obs)
        action, _states = model.predict(obs)
        # Take action
        obs, rewards, dones, info = env.step(action)

        # Return Data
        if i > steps-display_steps:
            print("reward=", rewards, "action to take=", action)
        # Check if done, then reset
        if dones == True:
            if i > steps-display_steps:
                env.render()
            print("Env complete! - - - - - - - - - - - -", "reward=", rewards)
            if rewards ==1:
                wins += 1
            elif rewards == -1:
                losses += 1
            else:
                halts += 1
            obs = env.reset()


    print(env.observation_space)
    print(env.action_space)
    print(env.action_space.sample())
    print("wins=", wins)
    print("losses=", losses)
    print("halts=", halts)
    if wins != 0:
        print("average steps per win=", steps/wins)
        print("success %=", str((wins/(wins+halts+losses))*100) + "%")

    #mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

    #print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    env.close()

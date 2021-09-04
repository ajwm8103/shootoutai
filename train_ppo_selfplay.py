import shootout
import os
import gym
import time
import random
import numpy as np


from stable_baselines.ppo1 import PPO1
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import logger
from stable_baselines.common.callbacks import EvalCallback

from shutil import copyfile # keep track of generations

LOGDIR = "models/ppo1_selfplay"

NUM_TIMESTEPS = int(1e9)
EVAL_FREQ = int(1e4)
EVAL_EPISODES = int(1e2)
BEST_THRESHOLD = 1.6

RENDER_MODE = True

class ShootoutSelfPlayEnv(shootout.ShootoutEnv):
	# wrapper over the normal single player env, but loads the best self play model
	def __init__(self):
		super(ShootoutSelfPlayEnv, self).__init__()
		self.policy = self
		self.best_model = None
		self.best_model_filename = None


	#sus = 0
	random_mode = 0
	def predict(self, obs): # the policy
		if self.best_model is None or self.random_mode <= 0.05:
			out = self.action_space.sample() # return a random action
			for i in range(len(out)):
				if i != 4:
					out[i] = round(out[i])
			#if self.sus % 2000 == 0:
			#print(f'blahblah {self.sus}, {self.timer}')
			#self.sus += 1
			return out
		elif self.random_mode <= 0.075:
			return np.array([0,0,0,0,0,0,0,0], dtype=np.float32)
		elif self.random_mode <= 0.1:
			return np.array([0,0,0,0,random.random(),0,0,1], dtype=np.float32)
		else:
			action, _ = self.best_model.predict(obs)
			return action
	def reset(self):
		# load model if it's there
		self.random_mode = random.random()
		modellist = [f for f in os.listdir(LOGDIR) if f.startswith("history")]
		modellist.sort()

		#andomlist = ["random/"+f for f in os.listdir(LOGDIR+"/random/") if f.startswith("history")]
		#randomlist.sort()
		if len(modellist) > 0:
			filename = os.path.join(LOGDIR, modellist[-1]) # the latest best model
			#filename = os.path.join(LOGDIR, randomlist[random.randint(0,len(randomlist)-1)] if self.random_mode <= 0.4 else modellist[-1]) # the latest best model
			if filename != self.best_model_filename:
				print("loading model as best: ", filename)
				self.best_model_filename = filename
				if self.best_model is not None:
					del self.best_model
				self.best_model = PPO1.load(filename, env=self)
		return super(ShootoutSelfPlayEnv, self).reset()

class SelfPlayCallback(EvalCallback):
	# hacked it to only save new version of best model if beats prev self by BEST_THRESHOLD score
	# after saving model, resets the best score to be BEST_THRESHOLD
	def __init__(self, *args, **kwargs):
		super(SelfPlayCallback, self).__init__(*args, **kwargs)
		self.best_mean_reward = BEST_THRESHOLD
		self.generation = 0
	def _on_step(self) -> bool:
		result = super(SelfPlayCallback, self)._on_step()
		if result and self.best_mean_reward > BEST_THRESHOLD:
			self.generation += 1
			env = ShootoutSelfPlayEnv()
			rollout(env, None)
			env.close()
			print("SELFPLAY: mean_reward achieved:", self.best_mean_reward)
			print("SELFPLAY: new best model, bumping up generation to", self.generation)
			source_file = os.path.join(LOGDIR, "best_model.zip")
			backup_file = os.path.join(LOGDIR, "history_"+str(self.generation).zfill(8)+".zip")
			copyfile(source_file, backup_file)
			self.best_mean_reward = BEST_THRESHOLD
		return result

def rollout(env, policy):
	""" play one agent vs the other in modified gym-style loop. """
	obs = env.reset()
	#env.random_mode = 0.35

	if policy == None:
		# load model if it's there
		modellist = [f for f in os.listdir(LOGDIR) if f.startswith("history")]
		modellist.sort()
		if len(modellist) > 0:
			filename = os.path.join(LOGDIR, modellist[-1]) # the latest best model
			if filename != None:
				print("loading model for rollout: ", filename)
				best_model_filename = filename
				policy = PPO1.load(filename, env=env)

				done = False
				total_reward = 0

				while not done:
					action, _states = policy.predict(obs)
					obs, reward, done, _ = env.step(action)

					total_reward += reward

					if RENDER_MODE:
						env.render()
						time.sleep(0.05)

				return total_reward
	return 0

def train():
	# train selfplay agent
	logger.configure(folder=LOGDIR)



	if False:
		# Load model from file
		# load model if it's there
		modellist = [f for f in os.listdir(LOGDIR) if f.startswith("history")]
		modellist.sort()
		if len(modellist) > 0:
			filename = os.path.join(LOGDIR, modellist[-1]) # the latest best model
		if filename != None:
			print("loading model: ", filename)
			best_model_filename = filename
			policy = PPO1.load(filename, env=env)
			model = policy
	else:

		# take mujoco hyperparams (but doubled timesteps_per_actorbatch to cover more steps.)
		model = PPO1(MlpPolicy, env, timesteps_per_actorbatch=4096, clip_param=0.2, entcoeff=0.0, optim_epochs=10,
					   optim_stepsize=3e-4, optim_batchsize=64, gamma=0.99, lam=0.95, schedule='linear', verbose=2)


	eval_callback = SelfPlayCallback(env,
	best_model_save_path=LOGDIR,
	log_path=LOGDIR,
	eval_freq=EVAL_FREQ,
	n_eval_episodes=EVAL_EPISODES,
	deterministic=False)

	model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)

	model.save(os.path.join(LOGDIR, "final_model")) # probably never get to this point.



	env.close()

if __name__=="__main__":

	env = ShootoutSelfPlayEnv()

	#rollout(env, None)
	train()

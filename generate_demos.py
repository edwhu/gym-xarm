import gymnasium as gym
import gym_xarm
import imageio
import numpy as np

debug = False
num_episodes = 100
render_mode = 'human' if debug else 'rgb_array'
env = gym.make("gym_xarm/PrivilegedXarmLift-v0", render_mode=render_mode)

demos = {
    'observations': {
        'pixels': [],
        'state': []
    },
    'next_observations': {
        'pixels': [],
        'state': []
    },
    'actions': [],
    'rewards': [],
    'dones': [],
}

action_noise = 0.5

for ep_idx in range(num_episodes):
    observation, info = env.reset()
    for k, v in observation.items():
        demos['observations'][k].append(v)
    debug and env.render()
    timestep = 0

    # first approach a point over the object.
    goal_pos = env.unwrapped.obj + np.array([0.0, 0, 0.1])
    eef_pos = env.unwrapped.eef
    while np.linalg.norm(eef_pos - goal_pos) > 0.025:
        pos_diff = (goal_pos - eef_pos) * 10
        pos_diff += np.random.normal(0, action_noise, size=pos_diff.shape)
        pos_diff =  np.clip(pos_diff, -1, 1)
        gripper_angle = -1.0
        action = np.concatenate([pos_diff, [gripper_angle]]).astype(np.float32)
        observation, reward, terminated, truncated, info = env.step(action)
        timestep += 1
        debug and env.render()
        eef_pos = env.unwrapped.eef
        for k, v in observation.items():
            demos['observations'][k].append(v)
        demos['actions'].append(action)
        demos['rewards'].append(reward)
        demos['dones'].append(terminated or truncated)

    # go down to the object
    goal_pos = env.unwrapped.obj
    eef_pos = env.unwrapped.eef
    while np.linalg.norm(eef_pos - goal_pos) > 0.025:
        pos_diff = (goal_pos - eef_pos) * 10
        pos_diff += np.random.normal(0, action_noise, size=pos_diff.shape)
        pos_diff =  np.clip(pos_diff, -1, 1)
        gripper_angle = -1.0
        action = np.concatenate([pos_diff, [gripper_angle]]).astype(np.float32)
        observation, reward, terminated, truncated, info = env.step(action)
        timestep += 1
        debug and env.render()
        eef_pos = env.unwrapped.eef
        for k, v in observation.items():
            demos['observations'][k].append(v)
        demos['actions'].append(action)
        demos['rewards'].append(reward)
        demos['dones'].append(terminated or truncated)

    # close the gripper
    for _ in range(10):
        action = np.array([0, 0, 0, 1.0], dtype=np.float32)
        observation, reward, terminated, truncated, info = env.step(action)
        timestep += 1
        debug and env.render()
        for k, v in observation.items():
            demos['observations'][k].append(v)
        demos['actions'].append(action)
        demos['rewards'].append(reward)
        demos['dones'].append(terminated or truncated)

    # lift the object up
    for i in range(100):
        action = np.array([0, 0, 1.0, 1.0], dtype=np.float32)
        observation, reward, terminated, truncated, info = env.step(action)
        timestep += 1
        debug and env.render()
        for k, v in observation.items():
            demos['observations'][k].append(v)
        demos['actions'].append(action)
        demos['rewards'].append(reward)
        demos['dones'].append(terminated or truncated)
        if terminated or truncated:
            print(f'Episode {ep_idx + 1} took {timestep} actions, success', info['is_success'])
            break
    
    for k, v in demos['observations'].items():
        demos['observations'][k] = v[:-1]
        demos['next_observations'][k] = v[1:]

env.close()

# convert the demos to numpy arrays
for k, v in demos.items():
    if isinstance(v, dict):
        for k2, v2 in v.items():
            demos[k][k2] = np.stack(v2, axis=0)
    else:
        demos[k] = np.stack(v, axis=0)

print('\nfinal demo dataset')
for k, v in demos.items():
    if isinstance(v, dict):
        for k2, v2 in v.items():
            print(k, k2, v2.shape)
    else:
        print(k, v.shape)

# store as a pickle file.
import pickle 
with open('gym_xarm_lift_demos.pkl', 'wb') as f:
    pickle.dump(demos, f)
from gymnasium.envs.registration import register

register(
    id="gym_xarm/XarmLift-v0",
    entry_point="gym_xarm.tasks:Lift",
    max_episode_steps=300,
    kwargs={"obs_type": "state"},
)


register(
    id="gym_xarm/PrivilegedXarmLift-v0",
    entry_point="gym_xarm.tasks:Lift",
    max_episode_steps=50,
    kwargs={"obs_type": "pixels_state"},
)

register(
    id="gym_xarm/PixelsXarmLift-v0",
    entry_point="gym_xarm.tasks:Lift",
    max_episode_steps=50,
    kwargs={"obs_type": "pixels"},
)
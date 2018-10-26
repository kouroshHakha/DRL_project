from gym.envs.registration import register

register(
    id='ckt-v0',
    entry_point='gym_ckt.envs:SweepCkt',
)

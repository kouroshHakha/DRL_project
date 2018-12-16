import gym
from gym.wrappers import Monitor
import time
import skvideo.io
from envs.goal_env_wrapper import GoalEnvWrapper

env = gym.make('FetchReach-v1')
env = GoalEnvWrapper(env)
env = Monitor(env, './video', force=True, mode='training',)
obs = env.reset()
start = time.time()
# writer = skvideo.io.FFmpegWriter('movie.mp4', inputdict={'-r': '50'}, outputdict={})
while True:
    # env.render()
    obs, r, done, info = env.step(env.action_space.sample())
    # frame = env.render(mode='rgb_array')
    # writer.writeFrame(frame)
    # print(frame)
    if done: break
# writer.close()
print("time:{}".format(time.time()-start))


# import os, glob
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import re
#
# def get_nums(files):
#
#     nums = []
#     for file in files:
#         m = re.search('(\d+)', file)
#         nums.append(int(m.group(0)))
#     return nums
#
# dir_name = 'photo_arrays/1'
# files = []
# for file in os.listdir(dir_name):
#     if file.endswith('.png'):
#         files.append(file)
#
# iter_nums = get_nums(files)
#
# indices = sorted(range(len(iter_nums)), key= lambda x: iter_nums[x])
# iter_nums = [iter_nums[index] for index in indices]
# files = [files[index] for index in indices]
#
# rows = int(np.sqrt(len(files)))
# cols = len(files) // rows
# f, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
# f.suptitle('Distribution of sub-goals (failed case) during training')
# for i, (ax, fname) in enumerate(zip(axes.ravel(), files)):
#     full_fname = os.path.join(dir_name, fname)
#     ax.set_title('iter {0}'.format(iter_nums[i]))
#     image = Image.open(full_fname).convert("L")
#     image_arr = np.asarray(image)
#     ax.imshow(image, cmap='gray')
#
# plt.show()
# plt.tight_layout()
# plt.subplots_adjust(top=0.90)
# f.savefig(os.path.join(logger.dir, 'prediction_{0:03d}.jpg'.format(r_num)), bbox_inches='tight')
"""
In this version, the MPC is adopted to control the rate adaptation, with the future bandwidth having been known in advance. So we call this version MPC-Oracal
"""
import numpy as np
from pruning_v2 import solving_opt

MPC_FUTURE_CHUNK_COUNT = 7
M_IN_K = 1000.0
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000000


class ABRExpert:
    ''' a MPC-based planning method to optimize the expected returns in adaptive video streaming, with the throughput dynamics being known in advance '''
    def __init__(self, abr_env, rebuf_p, smooth_p, mpc_horizon = MPC_FUTURE_CHUNK_COUNT, total_chunk_num = 48):
        self.env = abr_env
        self.rebuf_p = rebuf_p
        self.smooth_p = smooth_p
        self.mpc_horizon = mpc_horizon
        self.total_chunk_num = total_chunk_num
        self.video_chunk_remain = total_chunk_num
        self.time_stamp = 0
        self.start_buffer = 0
        self.last_bit_rate = DEFAULT_QUALITY
        self.bit_rate = DEFAULT_QUALITY

    def optimal_action(self):
        # future chunks length (try 4 if that many remaining)
        last_index = int(self.total_chunk_num - self.video_chunk_remain -1)
        future_chunk_length = self.mpc_horizon
        if (self.total_chunk_num - last_index < self.mpc_horizon ):
            future_chunk_length = self.total_chunk_num - last_index

        # planning for the optimal choice for next chunk
        opt_a = solving_opt(self.env, self.start_buffer, self.last_bit_rate, future_chunk_length, self.rebuf_p, self.smooth_p)
        return opt_a

    def step(self, action): # execute the action 
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_chunk_sizes, next_chunk_psnrs, \
                 end_of_video, video_chunk_remain, curr_chunk_sizes, curr_chunk_psnrs \
                    = self.env.get_video_chunk(action)

        self.time_stamp += delay  # in ms
        self.time_stamp += sleep_time  # in ms

        self.last_bit_rate = self.bit_rate
        self.bit_rate = action
        self.start_buffer = buffer_size

        self.video_chunk_remain = video_chunk_remain

        if end_of_video:
            self.time_stamp = 0
            self.last_bit_rate = DEFAULT_QUALITY

        return delay, sleep_time, buffer_size, rebuf, video_chunk_size, next_chunk_sizes, next_chunk_psnrs, end_of_video, video_chunk_remain, curr_chunk_sizes, curr_chunk_psnrs


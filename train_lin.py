import os
import numpy as np
import tensorflow as tf
import env_oracle as env
import time
import load_trace
from mpc_prunning import solving_log, solving_log_true_bw
# import pool
import libcomyco_lin as libcomyco

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

S_INFO = 6
S_LEN = 8
A_DIM = 6
LR_RATE = 1e-4
DEFAULT_QUALITY = 1
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
MODEL_TEST_INTERVAL = 10
QOE_METRIC = 'log'
# REBUF_PENALTY = 2.66 #4.3
REBUFF_PENALTY_LIN = 4.3
REBUFF_PENALTY_LOG = 2.66
SMOOTH_PENALTY = 1
MPC_FUTURE_CHUNK_COUNT = 7

RANDOM_SEED = 42
BUFFER_NORM_FACTOR = 10.0
M_IN_K = 1000.0
RAND_RANGE = 1000
TRAIN_TRACES = './bwsets/all/test_traces/' #_test
# TRAIN_TRACES = './cooked_test_traces/' #_test
LOG_FILE = './results/'
TEST_LOG_FOLDER = './test_results/'

# fixed to envivo
VIDEO_SIZE_FILE = './envivo/size/video_size_'
# VMAF = './envivo/vmaf/video'
CHUNK_TIL_VIDEO_END_CAP = 48.0


def loopmain(sess, actor):
    video_size = {}  # in bytes
    # vmaf_size = {}
    for bitrate in range(A_DIM):
        video_size[bitrate] = []
        # vmaf_size[bitrate] = []
        with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
            for line in f:
                video_size[bitrate].append(int(line.split()[0]))
        # with open(VMAF + str(A_DIM - bitrate)) as f:
        #     for line in f:
        #         vmaf_size[bitrate].append(float(line))
    all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(TRAIN_TRACES)
    net_env = env.Environment(all_cooked_time=all_cooked_time,
                            all_cooked_bw=all_cooked_bw)
    with open(LOG_FILE + 'agent', 'w') as log_file, open(LOG_FILE + 'log_test', 'w') as test_log_file:
        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY
        # last_chunk_vmaf = None

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        a_real_batch = [action_vec]
        r_batch = []

        entropy_record = []
        time_stamp = 0
        sess.run(tf.global_variables_initializer())
        
        saver = tf.train.Saver(max_to_keep=1000)
        epoch = 0
        while True:
            delay, sleep_time, buffer_size, rebuf, \
                video_chunk_size, next_video_chunk_sizes, \
                end_of_video, video_chunk_remain = \
                    net_env.get_video_chunk(int(bit_rate))

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            if QOE_METRIC == 'lin':
            # -- lin scale reward --
                REBUF_PENALTY = REBUFF_PENALTY_LIN
                reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                        - REBUF_PENALTY * rebuf \
                        - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                                VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
                # reward_max = 4.3
            else:
            # -- log scale reward --
                REBUF_PENALTY = REBUFF_PENALTY_LOG
                log_bit_rate = np.log(VIDEO_BIT_RATE[bit_rate] / float(VIDEO_BIT_RATE[0]))
                log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[0]))

                reward = log_bit_rate \
                        - REBUF_PENALTY * rebuf \
                        - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)

            r_batch.append(reward)

            last_bit_rate = bit_rate

            state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size) / \
                float(delay) / M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :A_DIM] = np.array(
                next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[5, -1] = np.minimum(video_chunk_remain,
                                      CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
            action_prob, bit_rate = actor.predict(
                np.reshape(state, (-1, S_INFO, S_LEN)))

            # net_env.get_optimal(float(last_chunk_vmaf))

            ##----------------------------MPC having known the future bandwidth------------------------
            last_index = int(CHUNK_TIL_VIDEO_END_CAP - video_chunk_remain - 1)
            future_horizon = MPC_FUTURE_CHUNK_COUNT
            if (CHUNK_TIL_VIDEO_END_CAP - 1 - last_index < MPC_FUTURE_CHUNK_COUNT):
                future_horizon = CHUNK_TIL_VIDEO_END_CAP - 1 - last_index
            start_buffer = buffer_size

            action_real = solving_log_true_bw(start_buffer, int(last_bit_rate), int(future_horizon), net_env, REBUF_PENALTY, SMOOTH_PENALTY, QOE_METRIC)
            # action_real = int(net_env.optimal)

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1

            action_real_vec = np.zeros(A_DIM)
            action_real_vec[action_real] = 1
            
            actor.submit(state, action_real_vec)
            actor.train()

            entropy_record.append(actor.compute_entropy(action_prob[0]))
            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write(str(time_stamp) + '\t' +
                           str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                           str(buffer_size) + '\t' +
                           str(rebuf) + '\t' +
                           str(video_chunk_size) + '\t' +
                           str(delay) + '\t' +
                           str(VIDEO_BIT_RATE[action_real]) + '\t' +
                           str(entropy_record[-1]) + '\t' +
                           str(reward) + '\n')
            log_file.flush()

            # report experience to the coordinator
            if end_of_video:
                del s_batch[:]
                del a_batch[:]
                del r_batch[:]
                del a_real_batch[:]
                #del d_batch[:]
                del entropy_record[:]

                # so that in the log we know where video ends
                log_file.write('\n')

            # store the state and action into batches
            if end_of_video:
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here
                # last_chunk_vmaf = None
                #chunk_index = 0

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1

                action_real_vec = np.zeros(A_DIM)
                action_real_vec[action_real] = 1

                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(action_vec)
                a_real_batch.append(action_real_vec)

                epoch += 1
                if epoch % MODEL_TEST_INTERVAL == 0:
                    # actor.save('models/nn_model_ep_' + \
                    #     str(epoch) + '.ckpt')
                    saver.save(sess, 'models/nn_model_ep_' + str(epoch) + '.ckpt')
                    os.system('python rl_test_lin.py ' + 'models/nn_model_ep_' + \
                        str(epoch) + '.ckpt')
                    # os.system('python plot_results.py >> results.log')

                    ## -------------------------record the results--------------------------------
                    rewards = []
                    test_log_files_ = os.listdir(TEST_LOG_FOLDER)
                    for test_log_file_ in test_log_files_:
                        reward = []
                        with open(TEST_LOG_FOLDER + test_log_file_, 'r') as f:
                            for line in f:
                                parse = line.split()
                                try:
                                    reward.append(float(parse[-1]))
                                except IndexError:
                                    break
                        rewards.append(np.sum(reward[1:]))

                    rewards = np.array(rewards)

                    rewards_min = np.min(rewards)
                    rewards_5per = np.percentile(rewards, 5)
                    rewards_mean = np.mean(rewards)
                    rewards_median = np.percentile(rewards, 50)
                    rewards_95per = np.percentile(rewards, 95)
                    rewards_max = np.max(rewards)

                    test_log_file.write(str(int(epoch)) + '\t' +
                                str(rewards_min) + '\t' +
                                str(rewards_5per) + '\t' +
                                str(rewards_mean) + '\t' +
                                str(rewards_median) + '\t' +
                                str(rewards_95per) + '\t' +
                                str(rewards_max) + '\n')
                    test_log_file.flush()
                    ## ------------------------------------------------------------------
            else:
                s_batch.append(state)

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1

                a_batch.append(action_vec)
                a_real_batch.append(action_vec)

def main():
    # create result directory
    if not os.path.exists(LOG_FILE):
        os.makedirs(LOG_FILE)
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    actor = libcomyco.libcomyco(sess,
            S_INFO=S_INFO, S_LEN=S_LEN, A_DIM=A_DIM,
            LR_RATE=LR_RATE)
    # modify for single agent
    loopmain(sess, actor)


if __name__ == '__main__':
    main()

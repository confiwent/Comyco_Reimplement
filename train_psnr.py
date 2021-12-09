import os
import argparse
import numpy as np
import tensorflow as tf
import fixed_env_real_bw as env_oracle
import load_trace
from MPC_expert import ABRExpert
# import pool
import libcomyco_lin as libcomyco

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

S_INFO = 7
S_LEN = 8
A_DIM = 6
LR_RATE = 1e-4
DEFAULT_QUALITY = 1
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
MODEL_TEST_INTERVAL = 10
REBUF_PENALTY = 8
SMOOTH_PENALTY = 0.5
MPC_FUTURE_CHUNK_COUNT = 7

RANDOM_SEED = 42
BUFFER_NORM_FACTOR = 10.0
DB_NORM_FACTOR = 25.0
M_IN_K = 1000.0
RAND_RANGE = 1000
TRAIN_TRACES = './envivo/traces/pre_webget_1608/cooked_traces/' #_test
# TRAIN_TRACES = './cooked_test_traces/' #_test
LOG_FILE = './results/'
TEST_LOG_FOLDER = './test_results/'

CHUNK_TIL_VIDEO_END_CAP = 149.0

parser = argparse.ArgumentParser(description='Comyco-Huang-JSAC20')
parser.add_argument('--Avengers', action='store_true', help='Use the video of Avengers')
parser.add_argument('--LasVegas', action='store_true', help='Use the video of LasVegas')
parser.add_argument('--Dubai', action='store_true', help='Use the video of Dubai')


def loopmain(sess, actor, summary_ops, summary_vars, writer, args):
    if args.Avengers:
        video = 'Avengers'
    elif args.LasVegas:
        video = 'LasVegas'
    elif args.Dubai:
        video = 'Dubai'
    
    video_size_file = './envivo/video_size/' + video + '/video_size_'
    video_psnr_file = './envivo/video_psnr/' + video + '/chunk_psnr'
    # -----------------------------initialize the environment----------------------------------------
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TRAIN_TRACES)
    train_env = env_oracle.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw, video_size_file= video_size_file, video_psnr_file=video_psnr_file)
    train_env.set_env_info(0, 0, 0, int(CHUNK_TIL_VIDEO_END_CAP), VIDEO_BIT_RATE, REBUF_PENALTY, SMOOTH_PENALTY)
    with open(LOG_FILE + 'agent', 'w') as log_file, open(LOG_FILE + 'log_test', 'w') as test_log_file:
        expert = ABRExpert(train_env, REBUF_PENALTY, SMOOTH_PENALTY, int(MPC_FUTURE_CHUNK_COUNT), int(CHUNK_TIL_VIDEO_END_CAP))
        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY
        last_quality = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[0]))

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        a_real_batch = [action_vec]
        r_batch = []

        ce_loss = []
        entropy_record = []
        time_stamp = 0
        sess.run(tf.global_variables_initializer())
        
        saver = tf.train.Saver(max_to_keep=1000)
        epoch = 0
        while True:
            delay, sleep_time, buffer_size, rebuf, \
                    video_chunk_size, next_video_chunk_sizes, \
                        next_video_chunk_psnrs, end_of_video, video_chunk_remain, \
                            _, curr_chunk_psnrs = expert.step(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            curr_quality = curr_chunk_psnrs[bit_rate]
            reward =  curr_quality \
                        - REBUF_PENALTY * rebuf \
                            - SMOOTH_PENALTY * np.abs(curr_quality - last_quality)
            last_quality = curr_quality

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
            state[5, :A_DIM] = np.array(next_video_chunk_psnrs) / DB_NORM_FACTOR
            state[6, -1] = np.minimum(video_chunk_remain,
                                      CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
            action_prob, bit_rate = actor.predict(
                np.reshape(state, (-1, S_INFO, S_LEN)))

            # net_env.get_optimal(float(last_chunk_vmaf))

            ##----------------------------MPC having known the future bandwidth------------------------
            # last_index = int(CHUNK_TIL_VIDEO_END_CAP - video_chunk_remain - 1)
            # future_horizon = MPC_FUTURE_CHUNK_COUNT
            # if (CHUNK_TIL_VIDEO_END_CAP - 1 - last_index < MPC_FUTURE_CHUNK_COUNT):
            #     future_horizon = CHUNK_TIL_VIDEO_END_CAP - 1 - last_index
            # start_buffer = buffer_size

            # action_real = solving_log_true_bw(start_buffer, int(last_bit_rate), int(future_horizon), net_env, REBUF_PENALTY, SMOOTH_PENALTY, QOE_METRIC)
            # action_real = int(net_env.optimal)
            action_real = expert.optimal_action()

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1

            action_real_vec = np.zeros(A_DIM)
            action_real_vec[action_real] = 1
            
            actor.submit(state, action_real_vec)
            batch_s, batch_a = actor.train()
            if batch_s.shape[0] > 0:
                loss = actor.loss(batch_s, batch_a)
                ce_loss.append(loss)
            entropy = actor.compute_entropy(action_prob[0])
            entropy_record.append(entropy)
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
            # store the state and action into batches
            if end_of_video:
                del s_batch[:]
                del a_batch[:]
                del r_batch[:]
                del a_real_batch[:]

                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here
                last_quality = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[0]))
                #chunk_index = 0

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1

                action_real_vec = np.zeros(A_DIM)
                action_real_vec[action_real] = 1

                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(action_vec)
                a_real_batch.append(action_real_vec)

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0] : np.mean(ce_loss),
                    summary_vars[1] : np.mean(entropy_record)
                })

                writer.add_summary(summary_str, epoch)
                writer.flush()

                del ce_loss[:]
                del entropy_record[:]

                # so that in the log we know where video ends
                log_file.write('\n')

                epoch += 1
                if epoch % MODEL_TEST_INTERVAL == 0:
                    # actor.save('models/nn_model_ep_' + \
                    #     str(epoch) + '.ckpt')
                    saver.save(sess, 'models/nn_model_ep_' + str(epoch) + '.ckpt')
                    os.system('python rl_test_lin_v2.py ' + 'models/nn_model_ep_' + \
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
                        rewards.append(np.mean(reward[1:]))

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
    args = parser.parse_args()
    # create result directory
    if not os.path.exists(LOG_FILE):
        os.makedirs(LOG_FILE)
    os.system("rm " + LOG_FILE + "*")
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    actor = libcomyco.libcomyco(sess,
            S_INFO=S_INFO, S_LEN=S_LEN, A_DIM=A_DIM,
            LR_RATE=LR_RATE)

    summary_ops, summary_vars = libcomyco.build_summaries()
    writer = tf.summary.FileWriter(LOG_FILE, sess.graph)  # training monitor
    # modify for single agent
    loopmain(sess, actor, summary_ops, summary_vars, writer, args)


if __name__ == '__main__':
    main()

import os
import sys
import argparse
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import tensorflow as tf
import load_trace
import libcomyco_lin as libcomyco
import fixed_env as env


S_INFO = 7  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
DB_NORM_FACTOR = 25.0
CHUNK_TIL_VIDEO_END_CAP = 149.0
M_IN_K = 1000.0
REBUF_PENALTY = 8.0
SMOOTH_PENALTY = 0.5
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
LOG_FILE = './test_results/log_test_cmc'
TEST_TRACES = './envivo/traces/pre_webget_1608/test_traces/'
NN_MODEL = sys.argv[1]

def main():

    video = 'Avengers'
    video_size_file = './envivo/video_size/' + video + '/video_size_'
    video_psnr_file = './envivo/video_psnr/' + video + '/chunk_psnr'

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    # -----------------------------initialize the environment----------------------------------------
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES)
    test_env = env.Environment(all_cooked_time=all_cooked_time,
                                all_cooked_bw=all_cooked_bw, all_file_names = all_file_names,
                                video_size_file = video_size_file, video_psnr_file=video_psnr_file)
    test_env.set_env_info(0, 0, 0, int(CHUNK_TIL_VIDEO_END_CAP), VIDEO_BIT_RATE, REBUF_PENALTY, SMOOTH_PENALTY)

    log_path = LOG_FILE + '_' + all_file_names[test_env.trace_idx]
    log_file = open(log_path, 'w')

    with tf.Session() as sess:
        actor = libcomyco.libcomyco(sess,
                S_INFO, S_LEN, A_DIM, LR_RATE = 1e-4)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        if NN_MODEL is not None:  # NN_MODEL is the path to file
            saver.restore(sess, NN_MODEL)
            print("Testing model restored.")

        time_stamp = 0

        bit_rate = DEFAULT_QUALITY
        last_bit_rate = DEFAULT_QUALITY
        last_quality = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[0]))

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []

        video_count = 0

        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, buffer_size, rebuf, \
                    video_chunk_size, next_video_chunk_sizes, \
                        next_video_chunk_psnrs, end_of_video, video_chunk_remain, \
                            _, curr_chunk_psnrs = test_env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            curr_quality = curr_chunk_psnrs[bit_rate]
            reward =  curr_quality \
                        - REBUF_PENALTY * rebuf \
                            - SMOOTH_PENALTY * np.abs(curr_quality - last_quality)
            last_quality = curr_quality
            r_batch.append(reward)
            last_bit_rate = bit_rate

            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write(str(time_stamp / M_IN_K) + '\t' +
                           str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                           str(buffer_size) + '\t' +
                           str(rebuf) + '\t' +
                           str(video_chunk_size) + '\t' +
                           str(delay) + '\t' +
                           str(reward) + '\n')
            log_file.flush()

            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[5, :A_DIM] = np.array(next_video_chunk_psnrs) / DB_NORM_FACTOR
            state[6, -1] = np.minimum(video_chunk_remain,
                                      CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

            action_prob, _ = actor.predict(
                np.reshape(state, (-1, S_INFO, S_LEN)))
            bit_rate = np.argmax(action_prob[0])

            s_batch.append(state)

            entropy_record.append(actor.compute_entropy(action_prob[0]))

            if end_of_video:
                log_file.write('\n')
                log_file.close()

                bit_rate = DEFAULT_QUALITY  # use the default action here
                last_bit_rate = DEFAULT_QUALITY
                last_quality = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[0]))

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1

                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(action_vec)
                entropy_record = []

                video_count += 1

                if video_count >= len(all_file_names):
                    break

                log_path = LOG_FILE + '_' + all_file_names[test_env.trace_idx]
                log_file = open(log_path, 'w')


if __name__ == '__main__':
    main()

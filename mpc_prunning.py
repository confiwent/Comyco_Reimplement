'''
MPC pruning for adaptive video streaming
'''

import numpy as np

MAX_REWARD = -10000000000
A_DIM = 6
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps


def solving_log(download_time_every_step, start_buffer, bit_rate, future_chunk_length, rebuf_penalty, smooth_penalty, qoe_metric):
    max_reward = MAX_REWARD
    reward_comparison = False
    send_data = 0
    parents_pool = [[0.0, start_buffer, int(bit_rate)]]
    for position in range(future_chunk_length):
        if position == future_chunk_length-1:
            reward_comparison = True
        children_pool = []
        for parent in parents_pool:
            action = 0
            curr_buffer = parent[1]
            last_quality = parent[-1]
            curr_rebuffer_time = 0
            chunk_quality = action
            download_time = download_time_every_step[position][chunk_quality]
            if ( curr_buffer < download_time ):
                curr_rebuffer_time += (download_time - curr_buffer)
                curr_buffer = 0.0
            else:
                curr_buffer -= download_time
            curr_buffer += 4

            # reward
            # bitrate_sum = VIDEO_BIT_RATE[chunk_quality]
            # smoothness_diffs = abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])
            if qoe_metric == 'log':
                rebuf_penalty = REBUFF_PENALTY_LOG
                bitrate_sum = np.log(VIDEO_BIT_RATE[chunk_quality] / float(VIDEO_BIT_RATE[0]))
                # smoothness_diffs = abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])
                smoothness_diffs = abs(np.log(VIDEO_BIT_RATE[chunk_quality] / float(VIDEO_BIT_RATE[0])) - np.log(
                    VIDEO_BIT_RATE[last_quality] / float(VIDEO_BIT_RATE[0])))
                reward = bitrate_sum - (rebuf_penalty * curr_rebuffer_time) - (
                        SMOOTH_PENALTY * smoothness_diffs)
            else:
                rebuf_penalty = REBUFF_PENALTY_LIN
                bitrate_sum = VIDEO_BIT_RATE[chunk_quality]
                smoothness_diffs = abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])
                reward = (bitrate_sum / 1000.) - (rebuf_penalty * curr_rebuffer_time) - (
                        SMOOTH_PENALTY * smoothness_diffs / 1000.)
            reward += parent[0]

            children = parent[:]
            children[0] = reward
            children[1] = curr_buffer
            children.append(action)
            children_pool.append(children)
            if (reward >= max_reward) and reward_comparison:
                if send_data > children[3] and reward == max_reward:
                    send_data = send_data
                else:
                    send_data = children[3]
                max_reward = reward

            # criterion terms
            # theta = SMOOTH_PENALTY * (VIDEO_BIT_RATE[action+1]/1000. - VIDEO_BIT_RATE[action]/1000.)
            if qoe_metric == 'log':
                rebuffer_term = rebuf_penalty * (max(download_time_every_step[position][action+1] - parent[1], 0) - max(download_time_every_step[position][action] - parent[1], 0))
                if (action + 1 <= parent[-1]):
                    High_Maybe_Superior = ((1.0 + 2 * rebuf_penalty) * (
                            np.log(VIDEO_BIT_RATE[action] / float(VIDEO_BIT_RATE[0])) - np.log(
                        VIDEO_BIT_RATE[action + 1] / float(VIDEO_BIT_RATE[0]))) + rebuffer_term < 0.0)
                else:
                    High_Maybe_Superior = (
                            (np.log(VIDEO_BIT_RATE[action] / float(VIDEO_BIT_RATE[0])) - np.log(
                                VIDEO_BIT_RATE[action + 1] / float(VIDEO_BIT_RATE[0]))) + rebuffer_term < 0.0)
            else:
                rebuffer_term = rebuf_penalty * (max(download_time_every_step[position][action+1] - parent[1], 0) - max(download_time_every_step[position][action] - parent[1], 0))
                if (action + 1 <= parent[-1]):
                    High_Maybe_Superior = ((1.0 + 2 * rebuf_penalty) * (VIDEO_BIT_RATE[action] / 1000. - VIDEO_BIT_RATE[
                        action + 1] / 1000.) + rebuffer_term < 0.0)
                else:
                    High_Maybe_Superior = ((VIDEO_BIT_RATE[action] / 1000. - VIDEO_BIT_RATE[
                        action + 1] / 1000.) + rebuffer_term < 0.0)



            # while rebuf_penalty*(download_time_every_step[position][action+1] - parent[1]) <= ((VIDEO_BIT_RATE[action+1]/1000. - VIDEO_BIT_RATE[action]/1000.)-(abs(VIDEO_BIT_RATE[action+1] - VIDEO_BIT_RATE[parent[-1]]) - abs(VIDEO_BIT_RATE[action] - VIDEO_BIT_RATE[parent[-1]]))/1000.):
            while High_Maybe_Superior:
                curr_buffer = parent[1]
                last_quality = parent[-1]
                curr_rebuffer_time = 0
                chunk_quality = action + 1
                download_time = download_time_every_step[position][chunk_quality]
                if ( curr_buffer < download_time ):
                    curr_rebuffer_time += (download_time - curr_buffer)
                    curr_buffer = 0
                else:
                    curr_buffer -= download_time
                curr_buffer += 4

                # reward
                # bitrate_sum = VIDEO_BIT_RATE[chunk_quality]
                # smoothness_diffs = abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])
                if qoe_metric == 'log':
                    # rebuf_penalty = REBUFF_PENALTY_LOG
                    bitrate_sum = np.log(VIDEO_BIT_RATE[chunk_quality] / float(VIDEO_BIT_RATE[0]))
                    # smoothness_diffs = abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])
                    smoothness_diffs = abs(np.log(VIDEO_BIT_RATE[chunk_quality] / float(VIDEO_BIT_RATE[0])) - np.log(
                        VIDEO_BIT_RATE[last_quality] / float(VIDEO_BIT_RATE[0])))
                    reward = bitrate_sum - (rebuf_penalty * curr_rebuffer_time) - (
                            SMOOTH_PENALTY * smoothness_diffs)
                else:
                    # rebuf_penalty = REBUFF_PENALTY_LIN
                    bitrate_sum = VIDEO_BIT_RATE[chunk_quality]
                    smoothness_diffs = abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])
                    reward = (bitrate_sum / 1000.) - (rebuf_penalty * curr_rebuffer_time) - (
                            SMOOTH_PENALTY * smoothness_diffs / 1000.)
                reward += parent[0]

                children = parent[:]
                children[0] = reward
                children[1] = curr_buffer
                children.append(chunk_quality)
                children_pool.append(children)
                if (reward >= max_reward) and reward_comparison:
                    if send_data > children[3] and reward == max_reward:
                        send_data = send_data
                    else:
                        send_data = children[3]
                    max_reward = reward

                action += 1
                if action + 1 == A_DIM:
                    break
                # criterion terms
                # theta = SMOOTH_PENALTY * (VIDEO_BIT_RATE[action+1]/1000. - VIDEO_BIT_RATE[action]/1000.)
                if qoe_metric == 'log':
                    rebuffer_term = rebuf_penalty * (max(download_time_every_step[position][action+1] - parent[1], 0) - max(download_time_every_step[position][action] - parent[1], 0))
                    if (action + 1 <= parent[-1]):
                        High_Maybe_Superior = ((1.0 + 2 * rebuf_penalty) * (
                                np.log(VIDEO_BIT_RATE[action] / float(VIDEO_BIT_RATE[0])) - np.log(
                            VIDEO_BIT_RATE[action + 1] / float(VIDEO_BIT_RATE[0]))) + rebuffer_term < 0.0)
                    else:
                        High_Maybe_Superior = (
                                (np.log(VIDEO_BIT_RATE[action] / float(VIDEO_BIT_RATE[0])) - np.log(
                                    VIDEO_BIT_RATE[action + 1] / float(VIDEO_BIT_RATE[0]))) + rebuffer_term < 0.0)
                else:
                    rebuffer_term = rebuf_penalty * (max(download_time_every_step[position][action+1] - parent[1], 0) - max(download_time_every_step[position][action] - parent[1], 0))
                    if (action + 1 <= parent[-1]):
                        High_Maybe_Superior = ((1.0 + 2 * rebuf_penalty)*(VIDEO_BIT_RATE[action]/1000. - VIDEO_BIT_RATE[action+1]/1000.) + rebuffer_term < 0.0)
                    else:
                        High_Maybe_Superior = ((VIDEO_BIT_RATE[action]/1000. - VIDEO_BIT_RATE[action+1]/1000.) + rebuffer_term < 0.0)

        parents_pool = children_pool

    return send_data


def solving_log_true_bw(start_buffer, bit_rate, future_chunk_length, env, rebuf_penalty, smooth_penalty, qoe_metric):
    max_reward = MAX_REWARD
    reward_comparison = False
    send_data = 0
    ## add last_time
    parents_pool = [[0.0,-1,-1,-1,-1, start_buffer, int(bit_rate)]]

    for position in range(future_chunk_length):
        if position == future_chunk_length-1:
            reward_comparison = True
        children_pool = []
        for parent in parents_pool:
            action = 0
            trace_idx = parent[1]
            video_chunk_counter = parent[2]
            mahimahi_ptr = parent[3]
            last_mahimahi_time = parent[4]
            curr_buffer = parent[5]
            last_quality = parent[-1]
            curr_rebuffer_time = 0
            chunk_quality = action
            # get download time with true bandwidth
            download_time, download_time_next, trace_idx_, video_chunk_counter_, mahimahi_ptr_, last_mahimahi_time_ = env.get_download_time(trace_idx, video_chunk_counter, mahimahi_ptr, last_mahimahi_time, chunk_quality)

            if ( curr_buffer < download_time ):
                curr_rebuffer_time += (download_time - curr_buffer)
                curr_buffer = 0.0
            else:
                curr_buffer -= download_time
            curr_buffer += 4

            # reward
            # bitrate_sum = VIDEO_BIT_RATE[chunk_quality]
            # smoothness_diffs = abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])
            if qoe_metric == 'log':
                # rebuf_penalty = REBUFF_PENALTY_LOG
                bitrate_sum = np.log(VIDEO_BIT_RATE[chunk_quality] / float(VIDEO_BIT_RATE[0]))
                # smoothness_diffs = abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])
                smoothness_diffs = abs(np.log(VIDEO_BIT_RATE[chunk_quality] / float(VIDEO_BIT_RATE[0])) - np.log(
                    VIDEO_BIT_RATE[last_quality] / float(VIDEO_BIT_RATE[0])))
                reward = bitrate_sum - (rebuf_penalty * curr_rebuffer_time) - (
                        smooth_penalty * smoothness_diffs)
            else:
                # rebuf_penalty = REBUFF_PENALTY_LIN
                bitrate_sum = VIDEO_BIT_RATE[chunk_quality]
                smoothness_diffs = abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])
                reward = (bitrate_sum / 1000.) - (rebuf_penalty * curr_rebuffer_time) - (
                        smooth_penalty * smoothness_diffs / 1000.)
            reward += parent[0]

            children = parent[:]
            children[0] = reward
            children[1] = trace_idx_
            children[2] = video_chunk_counter_
            children[3] = mahimahi_ptr_
            children[4] = last_mahimahi_time_
            children[5] = curr_buffer
            children.append(action)
            children_pool.append(children)
            if (reward >= max_reward) and reward_comparison:
                if send_data > children[7] and reward == max_reward:
                    send_data = send_data
                else:
                    send_data = children[7]
                max_reward = reward

            # criterion terms
            # theta = SMOOTH_PENALTY * (VIDEO_BIT_RATE[action+1]/1000. - VIDEO_BIT_RATE[action]/1000.)
            if qoe_metric == 'log':
                rebuffer_term = rebuf_penalty * (
                        max(download_time_next - parent[5], 0) - max(download_time - parent[5], 0))
                if (action + 1 <= parent[-1]):
                    High_Maybe_Superior = ((1.0 + 2 * rebuf_penalty) * (
                            np.log(VIDEO_BIT_RATE[action] / float(VIDEO_BIT_RATE[0])) - np.log(
                        VIDEO_BIT_RATE[action + 1] / float(VIDEO_BIT_RATE[0]))) + rebuffer_term < 0.0)
                else:
                    High_Maybe_Superior = (
                            (np.log(VIDEO_BIT_RATE[action] / float(VIDEO_BIT_RATE[0])) - np.log(
                                VIDEO_BIT_RATE[action + 1] / float(VIDEO_BIT_RATE[0]))) + rebuffer_term < 0.0)
            else:
                rebuffer_term = rebuf_penalty * (
                        max(download_time_next - parent[5], 0) - max(download_time - parent[5], 0))
                if (action + 1 <= parent[-1]):
                    High_Maybe_Superior = ((1.0 + 2 * rebuf_penalty) * (VIDEO_BIT_RATE[action] / 1000. - VIDEO_BIT_RATE[
                        action + 1] / 1000.) + rebuffer_term < 0.0)
                else:
                    High_Maybe_Superior = ((VIDEO_BIT_RATE[action] / 1000. - VIDEO_BIT_RATE[
                        action + 1] / 1000.) + rebuffer_term < 0.0)



            # while rebuf_penalty*(download_time_every_step[position][action+1] - parent[1]) <= ((VIDEO_BIT_RATE[action+1]/1000. - VIDEO_BIT_RATE[action]/1000.)-(abs(VIDEO_BIT_RATE[action+1] - VIDEO_BIT_RATE[parent[-1]]) - abs(VIDEO_BIT_RATE[action] - VIDEO_BIT_RATE[parent[-1]]))/1000.):
            while High_Maybe_Superior:
                trace_idx = parent[1]
                video_chunk_counter = parent[2]
                mahimahi_ptr = parent[3]
                last_mahimahi_time = parent[4]
                curr_buffer = parent[5]
                last_quality = parent[-1]
                curr_rebuffer_time = 0
                chunk_quality = action + 1



                download_time, download_time_next, trace_idx_, video_chunk_counter_, mahimahi_ptr_,last_mahimahi_time_= env.get_download_time(trace_idx, video_chunk_counter, mahimahi_ptr, last_mahimahi_time,chunk_quality)
                if ( curr_buffer < download_time ):
                    curr_rebuffer_time += (download_time - curr_buffer)
                    curr_buffer = 0
                else:
                    curr_buffer -= download_time
                curr_buffer += 4

                # reward
                # bitrate_sum = VIDEO_BIT_RATE[chunk_quality]
                # smoothness_diffs = abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])
                if qoe_metric == 'log':
                    # rebuf_penalty = REBUFF_PENALTY_LOG
                    bitrate_sum = np.log(VIDEO_BIT_RATE[chunk_quality] / float(VIDEO_BIT_RATE[0]))
                    # smoothness_diffs = abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])
                    smoothness_diffs = abs(np.log(VIDEO_BIT_RATE[chunk_quality] / float(VIDEO_BIT_RATE[0])) - np.log(
                        VIDEO_BIT_RATE[last_quality] / float(VIDEO_BIT_RATE[0])))
                    reward = bitrate_sum - (rebuf_penalty * curr_rebuffer_time) - (
                            smooth_penalty * smoothness_diffs)
                else:
                    # rebuf_penalty = REBUFF_PENALTY_LIN
                    bitrate_sum = VIDEO_BIT_RATE[chunk_quality]
                    smoothness_diffs = abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])
                    reward = (bitrate_sum / 1000.) - (rebuf_penalty * curr_rebuffer_time) - (
                            smooth_penalty * smoothness_diffs / 1000.)
                reward += parent[0]

                children = parent[:]
                children[0] = reward
                children[1] = trace_idx_
                children[2] = video_chunk_counter_
                children[3] = mahimahi_ptr_
                children[4] = last_mahimahi_time_
                children[5] = curr_buffer
                children.append(chunk_quality)
                children_pool.append(children)
                if (reward >= max_reward) and reward_comparison:
                    if send_data > children[7] and reward == max_reward:
                        send_data = send_data
                    else:
                        send_data = children[7]
                    max_reward = reward

                action += 1
                if action + 1 == A_DIM:
                    break
                # criterion terms
                # theta = SMOOTH_PENALTY * (VIDEO_BIT_RATE[action+1]/1000. - VIDEO_BIT_RATE[action]/1000.)
                if qoe_metric == 'log':
                    rebuffer_term = rebuf_penalty * (
                                max(download_time_next - parent[5], 0) - max(download_time - parent[5], 0))
                    if (action + 1 <= parent[-1]):
                        High_Maybe_Superior = ((1.0 + 2 * rebuf_penalty) * (
                                np.log(VIDEO_BIT_RATE[action] / float(VIDEO_BIT_RATE[0])) - np.log(
                            VIDEO_BIT_RATE[action + 1] / float(VIDEO_BIT_RATE[0]))) + rebuffer_term < 0.0)
                    else:
                        High_Maybe_Superior = (
                                (np.log(VIDEO_BIT_RATE[action] / float(VIDEO_BIT_RATE[0])) - np.log(
                                    VIDEO_BIT_RATE[action + 1] / float(VIDEO_BIT_RATE[0]))) + rebuffer_term < 0.0)
                else:
                    rebuffer_term = rebuf_penalty * (
                                max(download_time_next - parent[5], 0) - max(download_time - parent[5], 0))
                    if (action + 1 <= parent[-1]):
                        High_Maybe_Superior = ((1.0 + 2 * rebuf_penalty)*(VIDEO_BIT_RATE[action]/1000. - VIDEO_BIT_RATE[action+1]/1000.) + rebuffer_term < 0.0)
                    else:
                        High_Maybe_Superior = ((VIDEO_BIT_RATE[action]/1000. - VIDEO_BIT_RATE[action+1]/1000.) + rebuffer_term < 0.0)
        parents_pool = children_pool

    return send_data
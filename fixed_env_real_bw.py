import numpy as np

MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
RANDOM_SEED = 42
VIDEO_CHUNCK_LEN = 4000.0  # millisec, every time add this amount to buffer
BUFFER_THRESH = 60.0 * MILLISECONDS_IN_SECOND  # millisec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # millisec
PACKET_SIZE = 1500  # bytes
# NOISE_LOW = 0.3
# NOISE_HIGH = 1.7


class Environment:
    def __init__(self, all_cooked_time, all_cooked_bw, video_size_file, video_psnr_file, random_seed=RANDOM_SEED):
        assert len(all_cooked_time) == len(all_cooked_bw)

        np.random.seed(random_seed)

        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw

        self.video_chunk_counter = 0
        self.buffer_size = 0
        
        self.s_info = 17
        self.s_len = 10
        self.c_len = 3
        self.chunk_total_num = 48
        self.bitrate_version = [300, 750, 1200, 1850, 2850, 4300]
        self.br_dim = len(self.bitrate_version)
        self.rebuff_p = 2.66
        self.smooth_p = 1

        # pick a random trace file
        self.trace_idx = 0
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        self.mahimahi_start_ptr = 1
        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = 1
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        self.video_size = {}  # in bytes
        for bitrate in range(self.br_dim):
            self.video_size[bitrate] = []
            with open(video_size_file + str(bitrate)) as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))

        self.chunk_psnr = {} # video quality of chunks
        for bitrate in range(self.br_dim):
            self.chunk_psnr[bitrate] = []
            with open(video_psnr_file + str(bitrate)) as f:
                for line in f:
                    self.chunk_psnr[bitrate].append((float(line.split()[0])))

        self.total_chunk_num = len(self.video_size[0])

    def set_env_info(self, s_info, s_len, c_len, chunk_num, br_version, rebuff_p, smooth_p):
        self.s_info = s_info
        self.s_len = s_len
        self.c_len = c_len
        self.total_chunk_num = chunk_num
        self.bitrate_version = br_version
        self.br_dim = len(self.bitrate_version)
        self.rebuff_p = rebuff_p
        self.smooth_p = smooth_p

    def get_env_info(self):
        return self.s_info, self.s_len , self.c_len, self.chunk_total_num, self.bitrate_version, self.rebuff_p, self.smooth_p

    def get_video_chunk(self, quality):

        assert quality >= 0
        assert quality < self.br_dim

        video_chunk_size = self.video_size[quality][self.video_chunk_counter]

        # use the delivery opportunity in mahimahi
        delay = 0.0  # in ms
        video_chunk_counter_sent = 0  # in bytes

        while True:  # download video chunk over mahimahi
            throughput = self.cooked_bw[self.mahimahi_ptr] \
                         * B_IN_MB / BITS_IN_BYTE
            duration = self.cooked_time[self.mahimahi_ptr] \
                       - self.last_mahimahi_time

            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if video_chunk_counter_sent + packet_payload > video_chunk_size:

                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                  throughput / PACKET_PAYLOAD_PORTION
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                break

            video_chunk_counter_sent += packet_payload
            delay += duration
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.cooked_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        delay *= MILLISECONDS_IN_SECOND
        delay += LINK_RTT
        # delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)

        # rebuffer time
        rebuf = np.maximum(delay - self.buffer_size, 0.0)

        # update the buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)

        # add in the new chunk
        self.buffer_size += VIDEO_CHUNCK_LEN

        # sleep if buffer gets too large
        sleep_time = 0
        if self.buffer_size > BUFFER_THRESH:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size - BUFFER_THRESH
            sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * \
                         DRAIN_BUFFER_SLEEP_TIME
            self.buffer_size -= sleep_time

            while True:
                duration = self.cooked_time[self.mahimahi_ptr] \
                           - self.last_mahimahi_time
                if duration > sleep_time / MILLISECONDS_IN_SECOND:
                    self.last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND
                    break
                sleep_time -= duration * MILLISECONDS_IN_SECOND
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
                self.mahimahi_ptr += 1

                if self.mahimahi_ptr >= len(self.cooked_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr = 1
                    self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video
        return_buffer_size = self.buffer_size

        curr_chunk_sizes = []
        curr_chunk_psnrs = []
        for i in range(self.br_dim):
            curr_chunk_sizes.append(self.video_size[i][self.video_chunk_counter])
            curr_chunk_psnrs.append(self.chunk_psnr[i][self.video_chunk_counter])

        self.video_chunk_counter += 1
        video_chunk_remain = self.total_chunk_num - self.video_chunk_counter

        end_of_video = False
        if self.video_chunk_counter >= self.total_chunk_num:
            end_of_video = True
            self.buffer_size = 0
            self.video_chunk_counter = 0
            
            self.trace_idx += 1
            if self.trace_idx >= len(self.all_cooked_time):
                self.trace_idx = 0            

            self.cooked_time = self.all_cooked_time[self.trace_idx]
            self.cooked_bw = self.all_cooked_bw[self.trace_idx]

            # randomize the start point of the video
            # note: trace file starts with time 0
            self.mahimahi_ptr = self.mahimahi_start_ptr
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        next_video_chunk_sizes = []
        next_video_chunk_psnrs = []
        for i in range(self.br_dim):
            next_video_chunk_sizes.append(self.video_size[i][self.video_chunk_counter])
            next_video_chunk_psnrs.append(self.chunk_psnr[i][self.video_chunk_counter])

        return delay, \
            sleep_time, \
            return_buffer_size / MILLISECONDS_IN_SECOND, \
            rebuf / MILLISECONDS_IN_SECOND, \
            video_chunk_size, \
            next_video_chunk_sizes, \
            next_video_chunk_psnrs, \
            end_of_video, \
            video_chunk_remain, \
            curr_chunk_sizes, \
            curr_chunk_psnrs

    def get_curr_chunk_quality(self, trace_idx, video_chunk_counter, quality):
        if trace_idx == -1:
            video_chunk_counter = self.video_chunk_counter

        return self.chunk_psnr[quality][video_chunk_counter]

    def get_last_chunk_quality(self, trace_idx, video_chunk_counter, quality):
        if trace_idx == -1:
            video_chunk_counter = self.video_chunk_counter

        return self.chunk_psnr[quality][video_chunk_counter - 1]

    def get_download_time_upward(self,trace_idx, video_chunk_counter, mahimahi_ptr,last_mahimahi_time, chunk_quality):
        ## ---------------- compute last time ----------------------------------------------------
        if trace_idx == -1:
            trace_idx = self.trace_idx
            video_chunk_counter = self.video_chunk_counter
            mahimahi_ptr = self.mahimahi_ptr
            cooked_time = self.all_cooked_time[trace_idx]
            last_mahimahi_time = self.last_mahimahi_time
        ## ----------------- assign values ----------------------------------------------------

        cooked_bw = self.all_cooked_bw[trace_idx]
        cooked_time = self.all_cooked_time[trace_idx]

        ## ------------------- compute true bandwidth --------------------------------------------
        download_time = []
        for quality in range(chunk_quality, min(chunk_quality + 2, 6)):
            duration_all = 0
            video_chunk_counter_sent = 0  # in bytes
            video_chunk_size = self.video_size[quality][video_chunk_counter]
            mahimahi_ptr_tmp = mahimahi_ptr
            last_mahimahi_time_tmp = last_mahimahi_time

            while True:  # download video chunk over mahimahi
                throughput = cooked_bw[mahimahi_ptr_tmp] \
                             * B_IN_MB / BITS_IN_BYTE
                duration = cooked_time[mahimahi_ptr_tmp] \
                           - last_mahimahi_time_tmp

                packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

                if video_chunk_counter_sent + packet_payload > video_chunk_size:
                    fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                      throughput / PACKET_PAYLOAD_PORTION
                    last_mahimahi_time_tmp += fractional_time
                    duration_all += fractional_time
                    break
                video_chunk_counter_sent += packet_payload
                last_mahimahi_time_tmp = cooked_time[mahimahi_ptr_tmp]
                mahimahi_ptr_tmp += 1

                if mahimahi_ptr_tmp >= len(cooked_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    mahimahi_ptr_tmp = 1
                    last_mahimahi_time_tmp = 0
                duration_all += duration
            download_time.append(duration_all)
            if quality == chunk_quality:
                trace_idx_ = trace_idx
                video_chunk_counter_ = video_chunk_counter
                mahimahi_ptr_ = mahimahi_ptr_tmp
                last_mahimahi_time_ = last_mahimahi_time_tmp

        ## -------------------- test whether end of video ---------------------------------------------------
        video_chunk_counter_ += 1
        if video_chunk_counter_ >= self.total_chunk_num:

            video_chunk_counter_ = 0
            trace_idx_ += 1
            if trace_idx_ >= len(self.all_cooked_time):
                trace_idx_ = 0

            cooked_time = self.all_cooked_time[trace_idx_]
            cooked_bw = self.all_cooked_bw[trace_idx_]

            # randomize the start point of the video
            # note: trace file starts with time 0
            mahimahi_ptr_ = self.mahimahi_start_ptr
            last_mahimahi_time_ = cooked_time[mahimahi_ptr_ - 1]


        if len(download_time)==1:
            return download_time[0],0, trace_idx_, video_chunk_counter_, mahimahi_ptr_, last_mahimahi_time_
        else:
            return download_time[0],download_time[1], trace_idx_, video_chunk_counter_, mahimahi_ptr_, last_mahimahi_time_

    def get_download_time_downward(self,trace_idx, video_chunk_counter, mahimahi_ptr,last_mahimahi_time, chunk_quality):
        ## ---------------- compute last time ----------------------------------------------------
        if trace_idx == -1:
            trace_idx = self.trace_idx
            video_chunk_counter = self.video_chunk_counter
            mahimahi_ptr = self.mahimahi_ptr
            cooked_time = self.all_cooked_time[trace_idx]
            last_mahimahi_time = self.last_mahimahi_time
        ## ----------------- assign values ----------------------------------------------------

        cooked_bw = self.all_cooked_bw[trace_idx]
        cooked_time = self.all_cooked_time[trace_idx]

        ## ------------------- compute true bandwidth --------------------------------------------
        download_time = []
        for quality in range(chunk_quality, max(chunk_quality - 2, -1), -1):
            duration_all = 0
            video_chunk_counter_sent = 0  # in bytes
            video_chunk_size = self.video_size[quality][video_chunk_counter]
            mahimahi_ptr_tmp = mahimahi_ptr
            last_mahimahi_time_tmp = last_mahimahi_time

            while True:  # download video chunk over mahimahi
                throughput = cooked_bw[mahimahi_ptr_tmp] \
                             * B_IN_MB / BITS_IN_BYTE
                duration = cooked_time[mahimahi_ptr_tmp] \
                           - last_mahimahi_time_tmp

                packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

                if video_chunk_counter_sent + packet_payload > video_chunk_size:
                    fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                      throughput / PACKET_PAYLOAD_PORTION
                    last_mahimahi_time_tmp += fractional_time
                    duration_all += fractional_time
                    break
                video_chunk_counter_sent += packet_payload
                last_mahimahi_time_tmp = cooked_time[mahimahi_ptr_tmp]
                mahimahi_ptr_tmp += 1

                if mahimahi_ptr_tmp >= len(cooked_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    mahimahi_ptr_tmp = 1
                    last_mahimahi_time_tmp = 0
                duration_all += duration
            download_time.append(duration_all)
            if quality == chunk_quality:
                trace_idx_ = trace_idx
                video_chunk_counter_ = video_chunk_counter
                mahimahi_ptr_ = mahimahi_ptr_tmp
                last_mahimahi_time_ = last_mahimahi_time_tmp

        ## -------------------- test whether end of video ---------------------------------------------------
        video_chunk_counter_ += 1
        if video_chunk_counter_ >= self.total_chunk_num:

            video_chunk_counter_ = 0
            trace_idx_ += 1
            if trace_idx_ >= len(self.all_cooked_time):
                trace_idx_ = 0

            cooked_time = self.all_cooked_time[trace_idx_]
            cooked_bw = self.all_cooked_bw[trace_idx_]

            # randomize the start point of the video
            # note: trace file starts with time 0
            mahimahi_ptr_ = self.mahimahi_start_ptr
            last_mahimahi_time_ = cooked_time[mahimahi_ptr_ - 1]


        if len(download_time)==1:
            return download_time[0],0, trace_idx_, video_chunk_counter_, mahimahi_ptr_, last_mahimahi_time_
        else:
            return download_time[0],download_time[1], trace_idx_, video_chunk_counter_, mahimahi_ptr_, last_mahimahi_time_
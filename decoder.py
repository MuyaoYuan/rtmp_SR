import sys
from subprocess import Popen, PIPE
# import threading
import multiprocessing
import time
import numpy as np
import cv2 as cv

class Decoder(multiprocessing.Process):
    def __init__(self, url, config, new_frame_event, new_frame_lock, image_queue, debug=False, record=False):
        self.url = url
        self.config = config
        self.new_frame_event = new_frame_event
        self.new_frame_lock = new_frame_lock
        self.debug = debug
        self.record = record
        self.image_queue = image_queue
        self.commandline = 'ffmpeg\
                        -hide_banner\
                        -loglevel panic\
                        -protocol_whitelist \"file,http,https,rtp,udp,tcp,tls\"\
                        -i {url_}\
                        -f rawvideo\
                        -vf scale={width_}x{height_}:flags=lanczos\
                        -pix_fmt rgb24\
                        -r {fps_} -'\
                        .format(url_=self.url,
                                width_=self.config['width'],
                                height_=self.config['height'],
                                fps_=self.config['fps'])
        self.in_process = Popen(self.commandline, shell=True, stdout=PIPE, stderr=sys.stderr)
        # print(self.commandline)

        # Start the Process
        multiprocessing.Process.__init__(self)
        # 守护进程：设置为daemon的线程会随着主线程的退出而结束，而非daemon线程会阻塞主线程的退出。
        self.daemon = True
        self.start()

    def run(self):
        print('Opening input stream...')
        # 计算解码的fps
        if self.debug:
            is_first_frame = True
            frameCount = 0
            display_start_time = time.time()
        # 录制视频
        if self.record:
            out = cv.VideoWriter('out.flv', cv.VideoWriter_fourcc('F', 'L', 'V', '1'), 
                                    self.config['fps'], (self.config['width'], self.config['height']))
        while True:
            # 获取每一帧图像
            in_bytes = self.in_process.stdout.read(self.config['width'] * self.config['height'] * 3)
            if not in_bytes:
                if self.debug:
                    print('NO MORE BYTES IN PIPE')
                continue
            new_frame = np.frombuffer(in_bytes, np.uint8).reshape([self.config['height'], self.config['width'], 3])
            
            # 录制
            if self.record:
                bgr_frame = new_frame[:,:,::-1]
                out.write(bgr_frame)

            # 将新的一帧放入队列
            self.new_frame_lock.acquire()
            # print(id(self.image_queue))
            try:
                self.image_queue.put(new_frame,block=False)
            except:
                if self.debug:
                    print('队列已满')
            if self.debug:
                print("Queue in decoder, the length of queue:{}".format(self.image_queue.qsize()))
            self.new_frame_lock.release()
            # self.new_frame_event.set()

            # 计算解码的fps
            if self.debug:
                if is_first_frame:
                    is_first_frame = False
                    display_start_time = time.time()
                    frameCount = 0
                else:
                    frameCount += 1
                    rightNow = time.time()
                    fps = 1*frameCount / (rightNow - display_start_time)
                    print('PIPE FPS: ' + str(fps))

        if self.record:
            out.release()


if __name__ == '__main__':
    # url = 'kanna10.mp4'
    # url = 'HelloWorldRecorded.webm'
    # url = 'rtmp://localhost/tv/264663camera'
    url = 'test.sdp'
    # config = {
    #     'width':1920,
    #     'height':1080,
    #     'fps':60
    # }
    # config = {
    #         'width':640,
    #         'height':480,
    #         'fps':30
    # }
    config = {
            'width':160,
            'height':90,
            'fps':30
    }
    new_frame_lock = multiprocessing.Lock()
    new_frame_event = multiprocessing.Event()
    new_frame_event.clear()
    image_queue = multiprocessing.Queue() # 设置最大项数为10
    decoder = Decoder(url=url, config=config, new_frame_event=new_frame_event, new_frame_lock=new_frame_lock, image_queue=image_queue, debug=True ,record=True)
    while True:
        # print(image_queue.qsize())
        pass
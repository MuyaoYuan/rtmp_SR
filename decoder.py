import sys
from subprocess import Popen, PIPE
import threading
import time
import numpy as np
import cv2 as cv

class Decoder(threading.Thread):
    def __init__(self, url, config, new_frame_lock, image_queue, fps_count=False, record=False):
        self.url = url
        self.config = config
        self.new_frame_lock = new_frame_lock
        self.fps_count = fps_count
        self.record = record
        self.image_queue = image_queue
        self.commandline = 'ffmpeg -hide_banner -loglevel panic -i {url_} -f rawvideo -pix_fmt rgb24 -'.format(url_=self.url)
        self.in_process = Popen(self.commandline, shell=True, stdout=PIPE, stderr=sys.stderr)
        # print(self.commandline)

        # Start the thread
        threading.Thread.__init__(self)
        # 守护进程：设置为daemon的线程会随着主线程的退出而结束，而非daemon线程会阻塞主线程的退出。
        self.daemon = True
        self.start()

    def run(self):
        is_first_frame = True
        print('Opening input stream...')
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
                print('NO MORE BYTES')
                break
            new_frame = np.frombuffer(in_bytes, np.uint8).reshape([self.config['height'], self.config['width'], 3])
            
            # 录制
            if self.record:
                new_frame = new_frame[:,:,::-1]
                out.write(new_frame)

            # 将新的一帧放入队列，队列中永远只存在当前帧
            self.new_frame_lock.acquire()
            if len(self.image_queue) > 0:
                self.image_queue[0] = new_frame
            else:
                self.image_queue.append(new_frame)
            self.new_frame_lock.release()

            # 计算解码的fps
            if self.fps_count:
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
    url = 'kanna10.mp4'
    config = {
        'width':1920,
        'height':1080,
        'fps':60
    }
    new_frame_lock = threading.Lock()
    image_queue = list()
    decoder = Decoder(url=url, config=config, new_frame_lock=new_frame_lock, image_queue=image_queue, fps_count=True ,record=True)
    while True:
        # print(len(image_queue))
        pass
import sys
from subprocess import Popen, PIPE
import numpy as np
import cv2 as cv

class Decoder:
    def __init__(self, url, config, record=False, image_queue=None):
        self.url = url
        self.config = config
        self.commandline = 'ffmpeg -i {url_} -f rawvideo -pix_fmt rgb24 -'.format(url_=self.url)
        self.in_process = Popen(self.commandline, shell=True, stdout=PIPE, stderr=sys.stderr)
        print(self.commandline)
        # 录制视频
        if record:
            self.out = cv.VideoWriter('out.flv', cv.VideoWriter_fourcc('F', 'L', 'V', '1'), 
                                    self.config['fps'], (self.config['width'], self.config['height']))
        while True:
            # 获取每一帧图像
            in_bytes = self.in_process.stdout.read(self.config['width'] * self.config['height'] * 3)
            if not in_bytes:
                print('NO MORE BYTES')
                break
            new_frame = np.frombuffer(in_bytes, np.uint8).reshape([self.config['height'], self.config['width'], 3])
            if record:
                new_frame = new_frame[:,:,::-1]
                self.out.write(new_frame)
            if image_queue:
                image_queue.append(new_frame)


        if record:
            self.out.release()


if __name__ == '__main__':
    url = 'kanna10.mp4'
    config = {
        'width':1920,
        'height':1080,
        'fps':60
    }
    decoder = Decoder(url=url, config=config, record=True)
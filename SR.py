import torch
from torchvision.transforms.transforms import ToTensor
import numpy as np
import cv2 as cv
import sys
from subprocess import Popen, PIPE
import multiprocessing
import time

from model.ESPCN import ESPCN
from model.ESPCN_modified import ESPCN_modified
from model.ESPCN_multiframe import ESPCN_multiframe

from decoder import Decoder
from option import args

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class SR:
    def __init__(self, args, config, new_frame_event, new_frame_lock, image_queue, debug=False, record=False):
        self.args = args
        self.config = config
        self.new_frame_event = new_frame_event
        self.new_frame_lock = new_frame_lock
        self.image_queue = image_queue
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pipe_out = 'ffmpeg\
                            -hide_banner\
                            -f rawvideo\
                            -pixel_format rgb24\
                            -video_size {width_}x{height_}\
                            -r {fps_}\
                            -i - \
                            -r {fps_}\
                            -f flv\
                            rtmp://127.0.0.1/tv/test'\
                            .format(width_=self.config['width']*args.scale,
                                    height_=self.config['height']*args.scale,
                                    fps_=self.config['fps'])

        # select model
        self.model_name = args.model
        if(self.model_name == 'ESPCN' or self.model_name == 'ESPCN_modified'):
            self.mutilframe = False
        elif(self.model_name == 'ESPCN_multiframe' or self.model_name == 'ESPCN_multiframe2' or self.model_name == 'VESPCN'):
            self.mutilframe = True
            self.workQueue = list()
            self.queueIndex = 0

        if(self.model_name == 'ESPCN'):
            self.model = ESPCN(n_colors=args.n_colors, scale=args.scale).to(self.device)
        elif(self.model_name == 'ESPCN_modified'):
            self.model = ESPCN_modified(n_colors=args.n_colors, scale=args.scale).to(self.device)
        elif(self.model_name == 'ESPCN_multiframe'):
            self.model = ESPCN_multiframe(n_colors=args.n_colors, scale=args.scale, n_sequence=args.n_sequence).to(self.device)
        else:
            print('Please Enter Appropriate Model!!!')
        # save path
        self.save_path = 'trained_model/' + self.model_name + '/' + self.model_name + '.pkl'
        # model reload
        # print(self.save_path)
        self.model.load_state_dict(torch.load(self.save_path,map_location=self.device))
        self.transform = ToTensor()

        self.out_process = Popen(self.pipe_out, shell=True, stdin=PIPE, stderr=sys.stderr)
        
        if record:
            self.out = cv.VideoWriter('out.flv', cv.VideoWriter_fourcc('F', 'L', 'V', '1'), 
                                    self.config['fps'], (self.config['width'] * args.scale, self.config['height'] * args.scale))
        
        # ???????????????fps
        if debug:
            is_first_frame = True
            display_start_time = time.time()
            frameCount = 0

       
        # Event????????????????????????????????????????????????????????????????????????????????????????????????????????????wait???clear???set???isSet
        # set()????????????Event??????????????????????????????True
        # clear()????????????Event??????????????????????????????False
        # isSet()???Event???????????????isSet()?????????????????????????????????????????????????????????set()??????isSet()????????????True????????????clear()??????isSet()????????????False
        # wait()????????????????????????????????????True??????????????????????????????????????????????????????????????????False?????????wait()?????????????????????True????????????

        while True:
            # self.new_frame_event.wait()
            self.new_frame_lock.acquire()
            # print(id(self.image_queue))
            try:
                frame = self.image_queue.get(block=False)  # ????????????,??????????????????????????????
            except Exception:
                if debug:
                    print("????????????")
                # self.new_frame_event.clear()
                self.new_frame_lock.release()
                continue
            # self.new_frame_event.clear()
            self.new_frame_lock.release()

            if self.mutilframe:
                if(len(self.workQueue)<3):
                    self.workQueue.append(frame)
                    self.queueIndex = (self.queueIndex + 1) % args.n_sequence
                    continue
                else:
                    self.workQueue[self.queueIndex] = frame
                    self.queueIndex = (self.queueIndex + 1) % args.n_sequence

                    frames = list()
                    for i in range(args.n_sequence):
                        frames.append(self.transform(self.workQueue[(self.queueIndex + i) % args.n_sequence]).to(self.device))
                    
                    frames_in = torch.stack(frames, dim=0)
                    frames_in = frames_in.view(1, *frames_in.size())
                    frame_out = self.model(frames_in)
                    frame_process = frame_out_process(frame_out)
            else:
                frame_in = self.transform(frame)
                frame_in = frame_in.to(self.device)
                frame_in = frame_in.view(1, *frame_in.size())
                frame_out = self.model(frame_in)
                frame_process = frame_out_process(frame_out)

            
            if record:
                bgr_frame = frame_process[:,:,::-1]
                self.out.write(bgr_frame)

            self.out_process.stdin.write(frame_process.tobytes())
            
            # ???????????????fps
            if debug:
                if is_first_frame:
                    is_first_frame = False
                    display_start_time = time.time()
                    frameCount = 0
                else:
                    frameCount += 1
                    rightNow = time.time()
                    fps = 1*frameCount / (rightNow - display_start_time)
                    print('SR FPS: ' + str(fps))
        
        if record:
            self.out.release()

def frame_out_process(frame_out):
        frame_array = np.uint8(frame_out.cpu().detach().numpy()*255)
        frame_array = frame_array.transpose((0,2,3,1))
        frame_process = frame_array[0]
        return frame_process


if __name__ == '__main__':
    # url = 'kanna.mp4'
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
    image_queue = multiprocessing.Queue() # ?????????????????????10
    decoder = Decoder(url=url, config=config, new_frame_event=new_frame_event, new_frame_lock=new_frame_lock, image_queue=image_queue, debug=False, record=False)
    sr = SR(args=args, config=config, new_frame_event=new_frame_event, new_frame_lock=new_frame_lock, image_queue=image_queue, debug=True, record=False)
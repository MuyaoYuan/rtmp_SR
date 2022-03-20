import torch
from torchvision.transforms.transforms import ToTensor
import numpy as np
import cv2 as cv
import multiprocessing
import time

from model.ESPCN import ESPCN
from model.ESPCN_modified import ESPCN_modified
from model.ESPCN_multiframe import ESPCN_multiframe

from decoder import Decoder
from option import args

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class SR:
    def __init__(self, args, config, new_frame_event, new_frame_lock, image_queue, debug=False, record=False):
        self.args = args
        self.config = config
        self.new_frame_event = new_frame_event
        self.new_frame_lock = new_frame_lock
        self.image_queue = image_queue
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # select model
        self.model_name = args.model
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
        self.model.load_state_dict(torch.load(self.save_path))
        self.transform = ToTensor()
        if record:
            self.out = cv.VideoWriter('out.flv', cv.VideoWriter_fourcc('F', 'L', 'V', '1'), 
                                    self.config['fps'], (self.config['width'] * args.scale, self.config['height'] * args.scale))
        
        # 计算超分的fps
        if debug:
            is_first_frame = True
            display_start_time = time.time()
            frameCount = 0

       
        # Event对象用于线程间通信。用于主线程控制其他线程的执行，事件主要提供了四个方法wait、clear、set、isSet
        # set()：可设置Event对象内部的信号标志为True
        # clear()：可清除Event对象内部的信号标志为False
        # isSet()：Event对象提供了isSet()方法来判断内部的信号标志的状态。当使用set()后，isSet()方法返回True；当使用clear()后，isSet()方法返回False
        # wait()：该方法只有在内部信号为True的时候才会被执行并完成返回。当内部信号标志为False时，则wait()一直等待到其为True时才返回

        while True:
            # self.new_frame_event.wait()
            self.new_frame_lock.acquire()
            # print(id(self.image_queue))
            try:
                frame = self.image_queue.get(block=False)  # 取一个值,队列为空立马抛出异常
            except Exception:
                if debug:
                    print("队列已空")
                # self.new_frame_event.clear()
                self.new_frame_lock.release()
                continue
            # self.new_frame_event.clear()
            self.new_frame_lock.release()

            frame_in = self.transform(frame)
            frame_in = frame_in.to(self.device)
            frame_in = frame_in.view(1, *frame_in.size())
            frame_out = self.model(frame_in)
            frame_process = frame_out_process(frame_out)

            if record:
                record_frame = frame_process[:,:,::-1]
                self.out.write(record_frame)
            
            # 计算超分的fps
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
    url = 'HelloWorldRecorded.webm'
    # config = {
    #     'width':1920,
    #     'height':1080,
    #     'fps':60
    # }
    config = {
            'width':640,
            'height':480,
            'fps':30
    }
    new_frame_lock = multiprocessing.Lock()
    new_frame_event = multiprocessing.Event()
    new_frame_event.clear()
    image_queue = multiprocessing.Queue() # 设置最大项数为10
    decoder = Decoder(url=url, config=config, new_frame_event=new_frame_event, new_frame_lock=new_frame_lock, image_queue=image_queue, debug=False, record=False)
    sr = SR(args=args, config=config, new_frame_event=new_frame_event, new_frame_lock=new_frame_lock, image_queue=image_queue, debug=False, record=True)
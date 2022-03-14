import torch
from torchvision.transforms.transforms import ToTensor
import cv2 as cv

from model.ESPCN import ESPCN
from model.ESPCN_modified import ESPCN_modified
from model.ESPCN_multiframe import ESPCN_multiframe

from decoder import Decoder

class SR:
    def __init__(self, args, config, record=False, image_queue=None):
        self.args = args
        self.config = config
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

        if image_queue:
            while True:
                frame = image_queue.pop()
                frame_in = self.transform(frame)
                frame_in = frame_in.to(self.device)
                frame_in = frame_in.view(1, *frame_in.size())
                frame_out = self.model(frame_in)
                frame_process = frame_out_process(frame_out)
                if record:
                    frame_process = frame_process[:,:,::-1]
                    self.out.write(frame_process)
        if record:
            self.out.release()

        def frame_out_process(frame_out):
            frame_process = frame_out
            return frame_process

if __name__ == '__main__':
    pass
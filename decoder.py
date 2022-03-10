import cv2 as cv

class Decoder:
    def __init__(self, url):
        self.url = url
        self.cap = cv.VideoCapture(self.url)
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))

    def getFrame(self, doGet):
        while(self.cap.isOpened()):
            # 获取每一帧图像
            ret, frame = self.cap.read()
            # 如果获取成功显示图像
            if ret == True:
                doGet(frame)
            else:
                break
    
    def __del__(self):
        self.cap.release()
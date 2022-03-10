from decoder import Decoder
from SR import SR
from option import args

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

if args.task == 'single':
    pass

elif args.task == 'multithread':
    pass

elif args.task == 'multiprocess':
    pass

else:
    print('Please Enter Appropriate Task Type!!!')
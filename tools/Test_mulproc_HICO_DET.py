import os
import argparse
import glob
import _init_paths
import subprocess
import numpy as np
import sys
from ult.config import cfg
from queue import Queue, Empty
from threading import Thread
from sklearn.externals import joblib
def parse_args():
    parser = argparse.ArgumentParser(description='Test an pastanet on HICO DET')
    parser.add_argument('--gpu', dest='gpu',
            help='gpus to use',
            default='0', type=str)
    parser.add_argument('--iteration', dest='iteration',
            help='Number of iterations to load',
            default=10, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='multi_debug-2', type=str)
    parser.add_argument('--combine-only', dest='combine_only',
            help='only combine the results',
            action='store_true')
    args = parser.parse_args()
    return args
    
def enqueue_output(out, queue, i):
    for line in iter(out.readline, b''):
        queue.put((i, line))
    out.close()
    
def generate_test_list():
    cnt = 0
    with open('Data/test_list.txt','w') as f:
      for line in glob.iglob(cfg.DATA_DIR + '/' + 'hico_20160224_det/images/test2015/*.jpg'):
        f.write(line+'\n')
        cnt += 1
    return cnt
    
def combine_results(gpu_list, ranges, args):
    result_list = ['scores_P', 'scores_A', 'scores_L', 'bboxes', 'keys', 'hdet', 'odet']
    for each_result in result_list:
      locals()[each_result] = None
    for idx, each_gpu in enumerate(cfg.GPU_LIST):
       start, end = ranges[idx]
       output_file = cfg.ROOT_DIR + '/-Results/' + str(args.iteration) + '_' + args.model + '/' + 'range' + '_' + str(start) + '_' + str(end) + '/'
       for each_result in result_list:
         locals()[each_result+'_this_gpu'] = joblib.load(os.path.join(output_file, each_result+'.pkl'))
         if locals()[each_result] is None:
           locals()[each_result] = locals()[each_result+'_this_gpu']
         else:
           for i in range(80):
             locals()[each_result][i] = np.concatenate((locals()[each_result][i], locals()[each_result+'_this_gpu'][i]),axis=0)
    output_final_file = cfg.ROOT_DIR + '/-Results/' + str(args.iteration) + '_' + args.model + '/'
    for each_result in result_list:
      joblib.dump(locals()[each_result], os.path.join(output_final_file, each_result+'.pkl'))
      
if __name__ == '__main__':
    
    args = parse_args()
    cfg.GPU_LIST = list(map(lambda x: int(x),args.gpu.split(',')))
    gpu_num = len(cfg.GPU_LIST)
    cfg.MAX_POINTER = 1600000 // gpu_num
    test_length = generate_test_list()
    split_length = test_length // gpu_num
    ranges = [(split_length*gpu_idx, split_length*(gpu_idx+1)) for gpu_idx in range(gpu_num-1)]
    ranges.append((split_length*(gpu_num-1), test_length))
    
    if args.combine_only:
      combine_results(cfg.GPU_LIST, ranges, args)
      sys.exit()
    
    subprocess_env = os.environ.copy()
    q = Queue()
    processes = []
    for idx, each_gpu in enumerate(cfg.GPU_LIST):
      start, end = ranges[idx]
      idx_range = str(start) + ',' + str(end)
      i = each_gpu
      subprocess_env['CUDA_VISIBLE_DEVICES'] = str(each_gpu)
      cmd = 'python -u tools/Test_pasta_HICO_DET.py --iteration %d --model %s --range %s --max-pointer %d' % (args.iteration, args.model, idx_range, cfg.MAX_POINTER)
      p = subprocess.Popen(
            cmd,
            shell=True,
            env=subprocess_env,
            stdout=subprocess.PIPE,#subprocess_stdout,
            stderr=subprocess.STDOUT,
            bufsize=1,
            close_fds=True
        )
      t = Thread(target=enqueue_output, args=(p.stdout, q, i))
      t.daemon=True
      t.start()
      processes.append((i, p, start, end))
    alive_pool = np.ones(len(processes),dtype=np.int)
    # outputs = []
    while True:
      if alive_pool.sum() == 0:
        break
      try: 
        i, out = q.get_nowait()
      except Empty:
        pass
      else:
        print('gpu:', i, out.strip().decode('ascii'))
        
      for id, p, start, end in processes:
        if p.poll() is not None:
          alive_pool[id] = 0
          
    combine_results(cfg.GPU_LIST, ranges, args)
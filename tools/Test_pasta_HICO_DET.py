from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import tensorflow as tf
import numpy as np
import argparse
import pickle
import os

from ult.config import cfg
from models.test_Solver_HICO_DET_pasta import test_net
from networks.pasta_HICO_DET import ResNet50

def parse_args():
    parser = argparse.ArgumentParser(description='Test an pastanet on HICO DET')
    parser.add_argument('--gpu', dest='gpu',
            help='which gpu to use',
            default=0, type=int)
    parser.add_argument('--iteration', dest='iteration',
            help='Number of iterations to load',
            default=1800000, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='pasta_HICO_DET', type=str)
    parser.add_argument('--range', dest='range',
            help='test range of a single process',
            default='0,9658', type=str)
    parser.add_argument('--max-pointer', dest='max_pointer',
            help='max pointer',
            default=1600000, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    cfg.TRAIN_MODULE = 2
    args = parse_args()
    cfg.MAX_POINTER = args.max_pointer
    Test_RCNN = pickle.load( open( cfg.DATA_DIR + '/' + 'Test_all_part.pkl', "rb" ), encoding='bytes') # test detections

    test_list_path = cfg.DATA_DIR+'/'+'test_list.txt'
    test_list = [file.strip() for file in open(test_list_path,'r')]
    start, end = list(map(lambda x: int(x), args.range.split(',')))
    test_list_used = test_list[start:end]
    
    np.random.seed(cfg.RNG_SEED)
    # pretrain model
    weight = cfg.ROOT_DIR + '/Weights/' + args.model + '/HOI_iter_' + str(args.iteration) + '.ckpt'

    # output directory where the logs are saved
    print ('iter = ' + str(args.iteration) + ', path = ' + weight ) 
    output_parent = cfg.ROOT_DIR + '/-Results/' + str(args.iteration) + '_' + args.model + '/'
    output_file = cfg.ROOT_DIR + '/-Results/' + str(args.iteration) + '_' + args.model + '/' + 'range' + '_' + str(start) + '_' + str(end) + '/'
    
    os.makedirs(output_parent, exist_ok=True)
    os.makedirs(output_file, exist_ok=True)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.95
    tfconfig.gpu_options.allow_growth=True
    sess = tf.Session(config=tfconfig)
    net = ResNet50()
    with tf.device('/gpu:0'):
      with tf.name_scope('tower_0'):
        net.create_architecture(0, False)
    
        saver = tf.train.Saver()
        saver.restore(sess, weight)

        print('Pre-trained weights loaded.')
        sess.graph.finalize() 
        test_net(sess, net, Test_RCNN, output_file, test_list_used)
        
    sess.close()
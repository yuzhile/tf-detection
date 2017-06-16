from  __future__ import absolute_import
import os
import tensorflow as tf
import glob
from train import _draw_box
import time
import numpy as np
import cv2
from  config import *
from nets import * 

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint','./data/model_checkpoints/squeezeDet/model.ckpt-87000','Path to the model parameter file.')
tf.app.flags.DEFINE_string('input_path','./data/sample.png','Input image.')
tf.app.flags.DEFINE_string('out_dir','./data/out','Directory to dump output image.')
tf.app.flags.DEFINE_string('demo_net','squeezeDet','Neural net architecture.')
tf.app.flags.DEFINE_integer('gpu',0,'denate the gpu id.')

def image_demo():
    assert FLAGS.demo_net == 'squeezeDet' or FLAGS.demo_net == 'squeezeDet+',\
    'Selected nueral net architecture not supported: {}'.format(FLAGS.demo_net)
    with tf.Graph().as_default():
        #Load model
        if FLAGS.demo_net == 'squeezeDet':
            m = kitti_squeezeDet_config()
            m.BATCH_SIZE = 1
            m.LOAD_PRETRAINED_MODEL = False
            model = SqueezeDet(m,FLAGS.gpu)
        elif FLAGS.demo_net == 'squeezeDet+':
            m = kitti_squeezeDetPlus_config()
            m.BATCH_SIZE = 1
            m.LOAD_PRETRAINED_MODEL = False
        saver = tf.train.Saver(model.model_params)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            saver.restore(sess,FLAGS.checkpoint)
            for f in glob.iglob(FLAGS.input_path):
                print f
                im = cv2.imread(f)
                im = im.astype(np.float32,copy=False)
                im = cv2.resize(im,(m.IMAGE_WIDTH,m.IMAGE_HEIGHT))
                input_image = im - m.BGR_MEANS
                time_start = time.time()
                det_boxes, det_probs, det_class = sess.run([model.det_boxes, model.det_probs, model.det_class], feed_dict={model.image_input:[input_image]})
                print 'time run {} s'.format(time.time()-time_start)
                print 'boxes shape',det_boxes.shape,'probs shape',det_probs.shape,'class shape',det_class.shape
                final_boxes, final_probs, final_class = model.filter_prediction(det_boxes[0],det_probs[0],det_class[0])
                keep_idx = [idx for idx in range(len(final_probs)) \
                        if final_probs[idx] > m.PLOT_PROB_THRESH]
                final_boxes = [final_boxes[idx] for idx in keep_idx]
                final_probs = [final_probs[idx] for idx in keep_idx]
                final_class = [final_class[idx] for indx in keep_idx]
                print final_probs
                print final_class

                cls2clr = {
                        'car':(255,191,0),
                        'cyclist':(0,191,255),
                        'pedestrain':(255,0,191)
                        }
                _draw_box(im,final_boxes,
                        [m.CLASS_NAMES[idx]+': (% 2f)'% prob \
                                for idx, prob in zip(final_class,final_probs)],
                        cdict = cls2clr,
                        )
                file_name = os.path.split(f)[1]
                out_file_name = os.path.join(FLAGS.out_dir, 'out_'+file_name)
                cv2.imwrite(out_file_name,im)

def main(argv=None):
    print 'your are in main'
    if not tf.gfile.Exists(FLAGS.out_dir):
        tf.gfile.MakeDirs(FLAGS.out_dir)
    image_demo()

if __name__ == '__main__':
    tf.app.run()

from  __future__ import absolute_import
import tensorflow as tf
from  config import *
from nets import * 

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint','./data/model_checkpoints/squeezeDet/model.ckpt-87000','Path to the model parameter file.')
tf.app.flags.DEFINE_string('input_path','./data/sample.png','Input image.')
tf.app.flags.DEFINE_string('out_dir','./data/out','Directory to dump output image.')
tf.app.flags.DEFINE_string('demo_net','squeezeDet','Neural net architecture.')
tf.app.flags.DEFINE_integer('gpu',-1,'denate the gpu id.')

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
def main(argv=None):
    print 'your are in main'
    if not tf.gfile.Exists(FLAGS.out_dir):
        tf.gfile.MakeDirs(FLAGS.out_dir)
    image_demo()

if __name__ == '__main__':
    tf.app.run()

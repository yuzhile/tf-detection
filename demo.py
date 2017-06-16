from  __future__ import absolute_import
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint','./data/model.ckpt-87000','Path to the model parameter file.')
tf.app.flags.DEFINE_string('input_path','./data/sample.png','Input image.')
tf.app.flags.DEFINE_string('out_dir','./data/out','Directory to dump output image.')
tf.app.flags.DEFINE_string('demo_net','squeezeDet','Neural net architecture.')
def image_demo():
    pass 
def main(argv=None):
    print 'your are in main'
    if not tf.gfile.Exists(FLAGS.out_dir):
        tf.gfile.MakeDirs(FLAGS.out_dir)
    image_demo()

if __name__ == '__main__':
    tf.app.run()

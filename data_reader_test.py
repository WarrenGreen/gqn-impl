import tensorflow as tf

from data_reader import DataReader

root_path = '/mnt/es0/data/warren/gqn-impl/data'
scene_name = 'rooms_ring_camera'
data_reader = DataReader(dataset=scene_name, context_size=5, root=root_path)
data = data_reader.read(batch_size=12)
with tf.train.SingularMonitoredSession() as sess:
    d = sess.run(data)
    print(d)
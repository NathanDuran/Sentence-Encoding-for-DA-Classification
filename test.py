import tensorflow as tf
import pydot

s = tf.Session()
print(s.list_devices())

print("Test GPU available: "),
print(tf.test.is_gpu_available())

p = pydot.Dot()
p.create()
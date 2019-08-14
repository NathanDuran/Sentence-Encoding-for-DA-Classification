import tensorflow as tf

s = tf.Session()
print(s.list_devices())

print("Test GPU available: "),
print(tf.test.is_gpu_available())

# import pydot
# p = pydot.Dot()
# p.create()
import tensorflow as tf
import pydot

s = tf.Session()
print(s.list_devices())

print("Test GPU available: "),
print(tf.test.is_gpu_available())

p = pydot.Dot()
p.create()

# metric = False
# best_checkpoints = {}
# for i in range(3):
#     if metric:
#         best_checkpoints[str(i)] = float('inf')
#     else:
#         best_checkpoints[str(i)] = float('-inf')
#
# def keep_best(val, step):
#     if metric:
#         sorted_best = sorted(best_checkpoints.items(), reverse=True, key=lambda kv: kv[1])
#     else:
#         sorted_best = sorted(best_checkpoints.items(), reverse=False, key=lambda kv: kv[1])
#     print(sorted_best)
#     least_best_key = sorted_best[0][0]
#     least_best_val = sorted_best[0][1]
#     if(metric and val <= least_best_val) or (not metric and val >= least_best_val):
#             best_checkpoints.pop(least_best_key)
#             best_checkpoints[str(step)] = val
#
#
# keep_best(3.2, 5)
# print(best_checkpoints)
#
# keep_best(2.2, 10)
# print(best_checkpoints)
#
# keep_best(1.2, 15)
# print(best_checkpoints)
#
# keep_best(0.2, 20)
# print(best_checkpoints)
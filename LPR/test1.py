from hyperlpr import pipline as  pp
import cv2
import tensorflow as tf
#放在代码顶部的导入包的位置


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

# 自行修改文件名
image = cv2.imread("./car/demo1.png")
image,res  = pp.SimpleRecognizePlate(image)
print(res)

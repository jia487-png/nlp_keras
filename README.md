# nlp_keras
tensorflow 输入输出
# test1.py
~~~~
#测试压平层Flatten
import tensorflow as tf
from tensorflow import keras
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28))
])
print('原始数据，篇幅有限，每条训练数据包含28个一维张量，我们只显示第一条训练数据的前两个一维张量：')
print(x_train[0][0:2])
print('压平后的数据，篇幅有限，每条训练数据压平后是一个包含784个标量的一维张量，我们只显示前56个:')
print(model.predict(x_train)[0][0:56])

~~~~
# 运行结果
原始数据，篇幅有限，每条训练数据包含28个一维张量，我们只显示第一条训练数据的前两个一维张量：
[[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
压平后的数据，篇幅有限，每条训练数据压平后是一个包含784个标量的一维张量，我们只显示前56个:
I0000 00:00:1788863238.586076    2388 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 1s 743us/step 
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0.]

 # test2.py
~~~~

#测试第一个全连接层
import tensorflow as tf
from tensorflow import keras
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu')
])
print('原始数据，篇幅有限，每条训练数据包含28个一维张量，我们只显示第一条训练数据的前两个一维张量：：')
print(x_train[0][0:2])
print('压平后的数据再经过全连接层，每条训练数据经过压平和全连接后，是128个标量，由于全连接层初始权值是随机数，所以未经训练的全连接层输出的也是随机数')
print(model.predict(x_train)[0])


~~~~
# 运行结果
原始数据，篇幅有限，每条训练数据包含28个一维张量，我们只显示第一条训练数据的前两个一维张量：：
[[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
压平后的数据再经过全连接层，每条训练数据经过压平和全连接后，是128个标量，由于全连接层初始权值是随机数，所以未经训练的全连接层输出的也是随机数
WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 906us/step 
[0.00000000e+00 0.00000000e+00 7.84296112e+01 0.00000000e+00   
 2.27213726e+01 0.00000000e+00 0.00000000e+00 1.63286713e+02
 0.00000000e+00 4.95210342e+01 1.10840019e+02 0.00000000e+00
 5.10589828e+01 0.00000000e+00 0.00000000e+00 0.00000000e+00
 3.77890129e+01 0.00000000e+00 9.38267517e+01 7.55136490e+01
 1.69949684e+01 0.00000000e+00 1.66091671e+01 0.00000000e+00
 1.07372284e-01 9.19767151e+01 1.73156464e+02 0.00000000e+00
 8.54675064e+01 7.15516739e+01 0.00000000e+00 2.26640610e+02
 7.07904663e+01 7.26401978e+01 0.00000000e+00 0.00000000e+00
 1.68850464e+02 1.01055817e+02 1.31601822e+02 1.07672806e+01
 3.08792572e+01 0.00000000e+00 0.00000000e+00 0.00000000e+00
 1.40020885e+01 0.00000000e+00 0.00000000e+00 0.00000000e+00
 7.72875595e+01 2.03386215e+02 0.00000000e+00 1.22049179e+01
 0.00000000e+00 5.50873222e+01 0.00000000e+00 0.00000000e+00
 4.80590286e+01 1.26906509e+02 1.66742157e+02 5.57975998e+01
 0.00000000e+00 0.00000000e+00 4.30228081e+01 0.00000000e+00
 0.00000000e+00 0.00000000e+00 1.06354652e+02 0.00000000e+00
 0.00000000e+00 8.28407059e+01 1.81345505e+02 1.37466843e+02
 0.00000000e+00 0.00000000e+00 0.00000000e+00 2.36037960e+01
 7.18890686e+01 3.13927937e+01 0.00000000e+00 8.16130447e+01
 2.48566772e+02 0.00000000e+00 1.40816620e+02 0.00000000e+00
 0.00000000e+00 0.00000000e+00 0.00000000e+00 6.12326813e+00
 2.17623886e+02 0.00000000e+00 1.61633530e+02 2.06419434e+02
 1.24047134e+02 8.11820526e+01 0.00000000e+00 3.30166855e+01
 1.71102844e+02 9.43094711e+01 1.80210678e+02 0.00000000e+00
 1.23355652e+02 0.00000000e+00 0.00000000e+00 8.08158722e+01
 1.47287857e+02 0.00000000e+00 8.42554398e+01 0.00000000e+00
 1.44534744e+02 2.04223846e+02 0.00000000e+00 0.00000000e+00
 0.00000000e+00 0.00000000e+00 0.00000000e+00 1.49479828e+01
 0.00000000e+00 8.61713409e+01 0.00000000e+00 0.00000000e+00
 0.00000000e+00 1.45472839e+02 1.23778831e+02 1.86362000e+02
 0.00000000e+00 0.00000000e+00 2.57657074e+02 1.17346649e+01]

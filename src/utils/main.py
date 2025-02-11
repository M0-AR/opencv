import tensorflow as tf

# Check if TensorFlow is able to recognize your GPU
if tf.test.gpu_device_name():
    print(f'Default GPU Device: {tf.test.gpu_device_name()}')
else:
    print("Please install GPU version of TF")

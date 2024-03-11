import tensorflow as tf
print('TensorFlow version: {}'.format(tf.__version__))
from tfx import v1 as tfx
print('TFX version: {}'.format(tfx.__version__))

#print('Installing TensorFlow Data Validation')
#pip install --upgrade 'tensorflow_data_validation[visualization]<
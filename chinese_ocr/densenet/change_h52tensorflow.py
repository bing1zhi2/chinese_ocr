#-- coding:utf-8 --

from keras import backend as K
import os
import shutil
from imp import reload
import tensorflow as tf
from . import keys
from . import densenet
from keras.layers import Input
from keras.models import Model

from keras.utils import multi_gpu_model


K.set_learning_phase(0)

def preprocess_image(im):
    #等比例将图像高度缩放到32
    # im=tf.image.rgb_to_grayscale(im)
    im_shape = tf.shape(im)
    h=im_shape[1]
    w=im_shape[2]
    height=tf.constant(32,tf.int32)
    scale = tf.divide(tf.cast(h,tf.float32),32)
    width = tf.divide(tf.cast(w,tf.float32),scale)
    width =tf.cast(width,tf.int32)
    resize_image = tf.image.resize_images(im, [height,width], method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
    resize_image = tf.cast(resize_image, tf.float32) / 255 - 0.5
    width = tf.reshape(width, [1])
    height = tf.reshape(height, [1])
    im_info = tf.concat([height, width], 0)
    im_info = tf.concat([im_info, [1]], 0)
    im_info = tf.reshape(im_info, [1, 3])
    im_info = tf.cast(im_info, tf.float32)
    return resize_image,im_info


reload(densenet)

characters = keys.alphabet[:]
characters = characters[1:] + u'卍'
nclass = len(characters)
input = Input(shape=(32, None, 1), name='the_input')

# input = Input(tensor=resize_image)


y_pred= densenet.dense_cnn(input, nclass)
basemodel = Model(inputs=input, outputs=y_pred)
modelPath = os.path.join(os.getcwd(), './models/weights_densenet.h5')

# basemodel = multi_gpu_model(basemodel, gpus=8)

basemodel.load_weights(modelPath)
#model model_mulit_gpu
export_path = './model/1/'
if os.path.exists(export_path):
    shutil.rmtree(export_path)
version = 1
path='model'
K.set_learning_phase(0)
if not os.path.exists(path):
    os.mkdir(path)
export_path = os.path.join(tf.compat.as_bytes(path),tf.compat.as_bytes(str(version)))
builder = tf.saved_model.builder.SavedModelBuilder(export_path)
print('input info:', basemodel.input_names, '---', basemodel.input)
print('output info:', basemodel.output_names, '---', basemodel.output)
input_tensor = basemodel.input
output_tensor=basemodel.output

# tensor_info_input = tf.saved_model.utils.build_tensor_info(raw_image)

model_input = tf.saved_model.utils.build_tensor_info(input_tensor)
model_output = tf.saved_model.utils.build_tensor_info(output_tensor)

# im_info_output = tf.saved_model.utils.build_tensor_info(im_info)
# print('im_info_input tensor shape', im_info.shape)


prediction_signature = (
tf.saved_model.signature_def_utils.build_signature_def(
inputs={'images': model_input},
outputs={'prediction': model_output},
method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))


with K.get_session() as sess:
    builder.add_meta_graph_and_variables(
    sess=sess, tags=[tf.saved_model.tag_constants.SERVING],
    signature_def_map={'predict_images':prediction_signature,}
    )
    builder.save()
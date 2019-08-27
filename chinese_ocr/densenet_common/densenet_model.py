import multiprocessing
import os
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
from keras.layers.core import Lambda
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from PIL import Image

from chinese_ocr.train.train import random_uniform_num, get_session, get_model



# from config.train_config import DATASET_SOURCE1

def data_generator(dataset, config, maxlabellength=10):
    # batch_size=128,
    #                    imagesize=(32, 280)
    batch_size = config.BATCH_SIZE
    imagesize = (config.IMG_H, config.IMG_W)

    _imagefile = [info["id"] for info in dataset.image_info]
    x = np.zeros((batch_size, imagesize[0], imagesize[1], 1), dtype=np.float)
    labels = np.ones([batch_size, maxlabellength]) * 10000
    input_length = np.zeros([batch_size, 1])
    label_length = np.zeros([batch_size, 1])

    r_n = random_uniform_num(dataset.num_images)
    _imagefile = np.array(_imagefile)
    while 1:
        shufimagefile = dataset._image_ids[r_n.get(batch_size)]

        # print('shufimagefile:', shufimagefile)
        image_paths =[]
        for i, j in enumerate(shufimagefile):
            # image_id = dataset.map_source_image_id("Synthtext" + '.' + j)
            image_id = j

            image_path = dataset.image_info[image_id]["path"]
            image_paths.append(image_path)
            img1 = Image.open(image_path).convert('L')
            img = np.array(img1, 'f') / 255.0 - 0.5

            x[i] = np.expand_dims(img, axis=2)
            # print('imag:shape', img.shape)
            str = dataset.image_info[image_id]["label_list"]
            label_length[i] = len(str)

            if (len(str) <= 0):
                print("len < 0", j)
            input_length[i] = imagesize[1] // 8
            labels[i, :len(str)] = [int(k) - 1 for k in str]

        inputs = {'the_input': x,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  'img_paths': image_paths,
                  }
        outputs = {'ctc': np.zeros([batch_size])}
        yield (inputs, outputs)


class DensenetModel:
    def __init__(self, config, model_dir):
        """
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        self.config = config
        self.model_dir = model_dir



    def train(self, train_dataset, val_dataset, epochs, learning_rate=None,
              pre_train_path=None, checkpoint_path=None, no_augmentation_sources=None):
        assert train_dataset.num_classes == val_dataset.num_classes

        batch_size = self.config.BATCH_SIZE
        img_h = self.config.IMG_H

        nclass = train_dataset.num_classes

        K.set_session(get_session())
        # reload(densenet)
        basemodel, model = get_model(img_h, nclass)

        modelPath = './models/pretrain_model/keras.h5'
        if os.path.exists(modelPath):
            print("Loading model weights...")
            basemodel.load_weights(modelPath)
            print('done!')
        train_loader = data_generator(train_dataset, config=self.config)

        test_loader = data_generator(val_dataset, config=self.config)

        # checkpoint = ModelCheckpoint(filepath='./models/weights_densenet-{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', save_best_only=False, save_weights_only=True)
        checkpoint = ModelCheckpoint(filepath='./models/weights_densenet-{epoch:02d}-{val_loss:.2f}.h5',
                                     monitor='val_loss', save_best_only=False)

        # lr_schedule = lambda epoch: 0.0005 * 0.4 ** epoch
        # learning_rate = np.array([lr_schedule(i) for i in range(10)])
        # changelr = LearningRateScheduler(lambda epoch: float(learning_rate[epoch]))

        changelr = LearningRateScheduler(lambda epoch: 0.001 * 0.4 ** (epoch // 2))

        earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        tensorboard = TensorBoard(log_dir='./models/logs', write_graph=True)





        print('-----------Start training-----------')
        history = model.fit_generator(train_loader,
                            # steps_per_epoch = 3607567 // batch_size,
                            steps_per_epoch=train_dataset.num_images // batch_size,
                            epochs=epochs,
                            initial_epoch=0,
                            validation_data=test_loader,
                            # validation_steps = 36440 // batch_size,
                            validation_steps=val_dataset.num_images // batch_size,
                            callbacks=[checkpoint, earlystop, changelr, tensorboard])

        print(history)

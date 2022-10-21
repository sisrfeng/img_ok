from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import shutil

TypE = 'adult'
# TypE = 'violence'

model = load_model(f'final_{TypE}.h5')
test_data_dir = './imgs_{TypE}'
batch_size = 1
nb_samples = 1
SIZE = (224, 224)


test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(test_data_dir         ,
                                                  target_size = SIZE       ,
                                                  batch_size  = batch_size ,
                                                  class_mode  = 'binary'   ,
                                                  shuffle     = False      ,
                                                 )

result = model.predict_generator(test_generator, nb_samples // batch_size)

print('result是', result)


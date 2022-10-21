from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import shutil

TypE = 'adult'
# TypE = 'violence'

model = load_model(f'modelS/{TypE}.h5')
test_data_dir = f'./imgs_{TypE}'
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

correct_normal = 0
correct_bad    = 0

result = model.predict_generator(test_generator, nb_samples // batch_size)
prob = result[0][0]
print(f'是不良图片的概率',   round(prob , 3)  )
if  prob > 0.9:
    print(f'This img may be bad, ')
else:
    print(f'This img may be OK, { round(100 * result[0][0] , 1) }% sure')



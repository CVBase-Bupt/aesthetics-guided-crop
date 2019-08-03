try:
    import keras
    import tensorflow as tf
    import cv2
    import sys
    import os
    import numpy as np
    import time
except:
    print('Lack of corresponding dependencies.')

import keras.backend as K
if K.backend() != 'tensorflow':
    raise Exception("Only 'tensorflow' is supported as backend")

from utils import *
from keras.preprocessing.image import img_to_array, array_to_img
from PIL import Image, ImageDraw
import model as m

DRAW = True
def crop():

    #os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    img_path = sys.argv[1]
    if os.path.isfile(img_path):
        img_names = [img_path]
    elif os.path.isdir(img_path):
        img_names = os.listdir(img_path)
        img_names = [os.path.join(img_path, img_name) for img_name in img_names]
    else:
        raise Exception("The input path is not a image file or a image directory.")

    crop_img_save_path = 'crop_result' if len(sys.argv) <= 2 else sys.argv[-1]
    if not os.path.isdir(crop_img_save_path):
        os.makedirs(crop_img_save_path)

    #set model
    model_saliency = m.SaliencyUnet(state='test').BuildModel()
    model_regression = m.ofn_net(state='test').set_model()
    model_saliency.load_weights('model/saliency.h5')
    model_regression.load_weights('model/regression.h5')

    crop_regions = []
    saliency_regions = []

    total_time = saliency_time = crop_time = reg_time = 0
    for i in img_names:

        t_start = time.time()
        if not i.endswith(('.jpg', '.jpeg', '.png')):
            continue
        image_name = i.split('/')[-1]
        img = Image.open(i)
        img = img.convert('RGB')
        #img = get_shape(img, ratio, resize)

        image = np.asarray(img)

        h1, w1 = image.shape[0], image.shape[1]

        h2, w2 = (image.shape[0] / 16 + 1) * 16, (image.shape[1] / 16 + 1) * 16
        image = cv2.copyMakeBorder(image, top=0, bottom=h2 - h1, left=0, right=w2 - w1, borderType=cv2.BORDER_CONSTANT,
                                   value=0)
        image = image.astype('float32') / 255.0
        image = np.reshape(image, (1, image.shape[0], image.shape[1], 3))

        saliency_time_start = time.time()
        saliency_map = model_saliency.predict(image, batch_size=1, verbose=0)
        saliency_time_end = time.time()
        saliency_time += (saliency_time_end - saliency_time_start)
        saliency_map = saliency_map.reshape(h2, w2, 1)
        saliency_map = saliency_map * 255.0
        saliency_image = array_to_img(saliency_map)
        saliency_image = saliency_image.resize((w1, h1), Image.ANTIALIAS)
        pr = saliency_image.resize((224, 224), Image.ANTIALIAS)
        w_r, h_r = pr.size
        img_for_crop = np.asarray(pr).astype('float32')
        img_for_crop = img_for_crop.reshape((img_for_crop.shape[0], img_for_crop.shape[1]))
        crop_time_start = time.time()
        x, y, w, h = Minimum_Rectangle(img_for_crop, r=0.9)
        crop_time_end = time.time()
        crop_time += (crop_time_end - crop_time_start)

        x1 = x / float(h_r)
        x2 = (x + h) / float(h_r)
        y1 = y / float(w_r)
        y2 = (y + w) / float(w_r)
        if x2 > 1 or y2 > 1:
            x1, x2, y1, y2 = create_default_box(0.9, 'normal')
        saliency_box = [x1, x2, y1, y2]
        saliency_region = recover_from_normalization_with_order(w1 - 1, h1 - 1, saliency_box)
        saliency_img = img.crop(saliency_region)
        w4, h4 = saliency_img.size

        if w4 <= h4:
            saliency_img = saliency_img.resize((224, h4 * 224 / w4), Image.ANTIALIAS)
        else:
            saliency_img = saliency_img.resize((w4 * 224 / h4, 224), Image.ANTIALIAS)

        saliency_image = img_to_array(saliency_img)
        saliency_image = np.expand_dims(saliency_image, axis=0)
        saliency_image /= 255.0
        reg_time_start = time.time()
        offset = model_regression.predict(saliency_image, batch_size=1)[0]
        reg_time_end = time.time()
        reg_time += (reg_time_end - reg_time_start)
        #print i, saliency_box, offset
        final_region = add_offset(w, h, saliency_box, offset)

        final_region_to_file = ' '.join([image_name] + [str(u) for u in final_region])
        saliency_region_to_file = ' '.join([image_name] + [str(u) for u in saliency_box])
        crop_regions.append(final_region_to_file)
        saliency_regions.append(saliency_region_to_file)

        if DRAW:
            final_region = recover_from_normalization_with_order(w1, h1, final_region)

            pr = ImageDraw.Draw(img)
            # draw crop box on original image.

            pr.rectangle(final_region, None, 'yellow')
            pr.rectangle(saliency_region, None, 'blue')

            img.save(os.path.join(crop_img_save_path, image_name))

    total_time = saliency_time + crop_time + reg_time
    print('Average Total Time : %.3f s.' \
          'Average Saliency Time : %.3f s.' \
          'Average Crop Time : %.3f s.' \
          'Average Reg Time : %.3f s.' % (total_time / len(img_names), saliency_time / len(img_names),
                                                    crop_time / len(img_names), reg_time / len(img_names)))
    with open(os.path.join(crop_img_save_path, 'crop_region_coordinate.txt'), 'w') as f:
        f.write('\n'.join(crop_regions))
    with open(os.path.join(crop_img_save_path, 'saliency_region_coordinate.txt'), 'w') as f:
        f.write('\n'.join(saliency_regions))

if __name__ == '__main__':
    crop()

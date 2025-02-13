import time
import numpy as np
import cv2
import matplotlib.image as mpimg
import random
from tqdm import tqdm
import os
from PIL import Image



def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[80:-1, :, :] # remove the sky and the car front


def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (128, 128))


def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

def normalize(img):
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68
    return img

def denormalize(img):
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    return img

def preprocess(img_path):
    """
    Combine all preprocess functions into one
    """
    image = np.asarray(Image.open(img_path))
    if image.shape[0] == 128:
        image = normalize(np.float32(image))
    else:
        image = crop(image)
        image = resize(np.array(image)) # (h,w,3)
        image = normalize(np.float32(image))
    return image

def data_generator(xs, ys, batch_size=64, mode=None):
    gen_state = 0
    while 1:
        if gen_state + batch_size > len(xs):
            paths = xs[gen_state: len(xs)]
            y = ys[gen_state: len(xs)]
            X = [preprocess(x) for x in paths]
            gen_state = 0
        else:
            paths = xs[gen_state: gen_state + batch_size]
            y = ys[gen_state: gen_state + batch_size]
            X = [preprocess(x) for x in paths]
            gen_state += batch_size
            if gen_state == len(xs):
                gen_state = 0
        yield np.array(X), np.array(y)

def val_generator(xs, ys, batch_size=64, mode=None):
    gen_state = 0
    while 1:
        if gen_state == len(xs):
            return
        elif gen_state < len(xs) and (gen_state + batch_size) > len(xs):
            paths = xs[gen_state: len(xs)]
            y = ys[gen_state: len(xs)]
            X = [preprocess(x) for x in paths]
            yield np.array(X), np.array(y)
            return
        else:
            paths = xs[gen_state: gen_state + batch_size]
            y = ys[gen_state: gen_state + batch_size]
            X = [preprocess(x) for x in paths]
            gen_state += batch_size
            yield np.array(X), np.array(y)


def load_train_data(path='/data/udacity_output/Ch2_002/', batch_size=64, shape=(100, 100)):
    xs = []
    ys = []
    start_load_time = time.time()
    with open(path + 'interpolated.csv', 'r') as f:
        for i, line in enumerate(f):
            if i == 0 or 'center' not in line.split(',')[5]:
                continue
            xs.append(path + line.split(',')[5])

            regressed_val = -1 * float(line.split(',')[6])
            if regressed_val > 0.15:
                label = 2 # steer right
            elif regressed_val < -0.15:
                label = 1 # steer left
            else:
                label = 0 # go straight
            ys.append(label)

    # shuffle list of images
    c = list(zip(xs, ys))
    random.shuffle(c)
    xs, ys = zip(*c)

    train_xs = xs
    train_ys = ys

    train_generator = data_generator(train_xs, train_ys,
                                     batch_size=batch_size)

    return train_generator, (train_xs, train_ys)

def load_val_data(path='/data/udacity_output/Ch2_002/', batch_size=64, shape=(100, 100), start=None, end=None):
    xs = []
    ys = []
    start_load_time = time.time()
    gen_img_folder = path+'/center' #totl 33808 center imgs
    img_paths = ['center/'+img for img in os.listdir(gen_img_folder) if img.endswith(".jpg")][start:end]

    with open(path + 'interpolated.csv', 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            filename = line.split(',')[5]
            if filename in img_paths:
                xs.append(path + filename)
                # change regressed to classification with threshold = 0.15
                regressed_val = -1 * float(line.split(',')[6])
                if regressed_val > 0.15:
                    label = 2  # steer right
                elif regressed_val < -0.15:
                    label = 1  # steer left
                else:
                    label = 0  # go straight
                ys.append(label)
    # shuffle list of images
    c = list(zip(xs, ys))
    random.shuffle(c)
    xs, ys = zip(*c)

    train_xs = xs
    train_ys = ys

    return train_xs, train_ys


def load_test_data(path='/data/udacity_output/testing/', batch_size=64, shape=(100, 100)):
    xs = []
    ys = []
    start_load_time = time.time()
    with open(path + 'final_example.csv', 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            xs.append(path + 'center/' + line.split(',')[0] + '.jpg')

            regressed_val = -1 * float(line.split(',')[1])
            if regressed_val > 0.15:
                label = 2  # steer right
            elif regressed_val < -0.15:
                label = 1  # steer left
            else:
                label = 0  # go straight
            ys.append(label)
    # shuffle list of images
    c = list(zip(xs, ys))
    random.shuffle(c)
    xs, ys = zip(*c)
    train_xs = xs
    train_ys = ys
    return train_xs, train_ys

def DaveDataset(batch_size=64, no_generator=False):
    xs = []
    ys = []
    steer = []
    # 45406 images
    with open("/data/dave_test/driving_dataset/data.txt") as f:
        for line in tqdm(f):
            xs.append("/data/dave_test/driving_dataset/" + line.split()[0])
            # the paper by Nvidia uses the inverse of the turning radius,
            # but steering wheel angle is proportional to the inverse of turning radius
            # so the steering wheel angle in radians is used as the output
            steering_angle = float(line.split()[1]) * 3.14159265 / 180
            steer.append(steering_angle)
            if steering_angle > 0.15:
                label = 2  # steer right
            elif steering_angle < -0.15:
                label = 1  # steer left
            else:
                label = 0  # go straight
            ys.append(label)

    # shuffle list of images
    c = list(zip(xs, ys))
    random.shuffle(c)
    xs, ys = zip(*c)
    train_xs = xs[:int(len(xs) * 0.6)]
    train_ys = ys[:int(len(xs) * 0.6)]

    val_xs = xs[-int(len(xs) * 0.4):-int(len(xs) * 0.2)]
    val_ys = ys[-int(len(xs) * 0.4):-int(len(xs) * 0.2)]

    test_xs = xs[-int(len(xs) * 0.2):]
    test_ys = ys[-int(len(xs) * 0.2):]

    train_generator = data_generator(train_xs, train_ys,
                                     batch_size=batch_size)
    if no_generator:
        return train_xs, train_ys, val_xs, val_ys, test_xs, test_ys, np.array(steer)
    return train_generator, val_xs, val_ys, test_xs, test_ys, np.array(steer)


def SingleDaveDataset(input_file, path):
    xs = []
    ys = []
    steer = []
    with open(input_file) as f:
        for line in tqdm(f):
            xs.append(path + line.split()[0])
            steering_angle = float(line.split()[1]) * 3.14159265 / 180
            steer.append(steering_angle)
            if steering_angle > 0.15:
                label = 2
            elif steering_angle < -0.15:
                label = 1
            else:
                label = 0
            ys.append(label)
    return xs, ys, np.array(steer)


def LabeledDaveDataset(input_file, path):
    xs = []
    ys = []
    with open(input_file) as f:
        for line in tqdm(f):
            xs.append(path + line.split()[0])
            ys.append(int(line.split()[1]))
    return xs, ys

if __name__ == '__main__':
    train_generator, val_xs, val_ys, test_xs, test_ys, steering_values = DaveDataset(64)


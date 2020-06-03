"""Transforms that are aimed to create realistic weather conditions augmentations"""
# adapted from: https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library

import cv2
import numpy as np
from torchvision.transforms import ToPILImage
toPIL = ToPILImage()


def hls(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)


def rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_HLS2RGB)


# RAIN
def generate_random_lines(imshape, slant, drop_length):
    drops = []
    area = imshape[0] * imshape[1]
    no_of_drops = area // 600

    for i in range(no_of_drops):  ## If You want heavy rain, try increasing this
        if slant < 0:
            x = np.random.randint(slant, imshape[1])
        else:
            x = np.random.randint(0, imshape[1] - slant)
        y = np.random.randint(0, imshape[0] - drop_length)
        drops.append((x, y))

    return drops, drop_length


def rain_process(image, slant, drop_length, drop_color, drop_width, rain_drops):
    imshape = image.shape
    image_t = image.copy()
    for rain_drop in rain_drops:
        cv2.line(
            image_t,
            (rain_drop[0], rain_drop[1]),
            (rain_drop[0] + slant, rain_drop[1] + drop_length),
            drop_color, drop_width
        )
    image = cv2.blur(image_t, (4, 4))  # rainy view are blurry
    return image


def add_rain(img_data, slant):  # (-20 <= slant <= 20)
    drop_width = 1
    drop_length = 15
    drop_color = (200, 200, 200)  # (200, 200, 200) a shade of gray

    image_RGB = []
    imshape = img_data[0].shape

    rain_drops, drop_length = generate_random_lines(imshape, slant, drop_length)
    for img in img_data:
        output = rain_process(img, slant, drop_length, drop_color, drop_width, rain_drops)
        image_RGB.append(output)

    return image_RGB


# SNOW
def snow_process(image, snow_point):
    image_HLS = hls(image)  # Conversion to HLS
    image_HLS = np.array(image_HLS, dtype=np.float32)
    brightness_coefficient = 2
    image_HLS[:, :, 1][image_HLS[:, :, 1] < snow_point] = \
        image_HLS[:, :, 1][image_HLS[:, :, 1] < snow_point] * brightness_coefficient  # scale pixel values up for channel 1(Lightness)
    image_HLS[:, :, 1][image_HLS[:, :, 1] > 255] = 255  # Sets all values above 255 to 255
    image_HLS = np.array(image_HLS, dtype=np.uint8)
    image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)  # Conversion to RGB

    return image_RGB


def add_snow(img_data, snow_coeff):  # (0<=snow_coeff<=0.5) increase this for more snow
    snow_coeff *= 255 / 2
    snow_coeff += 255 / 3
    image_RGB = [snow_process(img, snow_coeff) for img in img_data]

    return image_RGB


def generate_random_blur_coordinates(imshape, hw):
    blur_points = []
    midx = imshape[1] // 2 - 2 * hw
    midy = imshape[0] // 2 - hw
    index = 1
    while (midx > -hw or midy > -hw):
        for i in range(hw // 10 * index):
            x = np.random.randint(midx, imshape[1] - midx - hw)
            y = np.random.randint(midy, imshape[0] - midy - hw)
            blur_points.append((x, y))
        midx -= 3 * hw * imshape[1] // sum(imshape)
        midy -= 3 * hw * imshape[0] // sum(imshape)
        index += 1
    return blur_points


def add_blur(image, x, y, hw, fog_coeff):
    overlay = image.copy()
    output = image.copy()
    alpha = 0.15 * fog_coeff
    rad = hw // 2
    point = (x + hw // 2, y + hw // 2)
    cv2.circle(overlay, point, int(rad), (255, 255, 255), -1)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output


def add_fog(img_data, fog_coeff):  # (0.3<=fog_coeff<=0.7)
    image_RGB = []
    imshape = img_data[0].shape
    hw = int(imshape[1] // 3 * fog_coeff)
    haze_list = generate_random_blur_coordinates(imshape, hw)

    for img in img_data:
        ## adding all shadow polygons on empty mask, single 255 denotes only red channel
        for haze_points in haze_list:
            img = add_blur(img, haze_points[0], haze_points[1], hw, fog_coeff)

        img = cv2.blur(img, (hw // 10, hw // 10))
        image_RGB.append(img)

    return image_RGB


# Motion speed
def apply_motion_blur(image, count):
    image_t = image.copy()
    imshape = image_t.shape
    size = 5
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size

    i = imshape[1] * 3 // 4 - 10 * count
    while (i < imshape[1] and i > 0):
        image_t[:, i:, :] = cv2.filter2D(image_t[:, i:, :], -1, kernel_motion_blur)
        image_t[:, :imshape[1] - i, :] = cv2.filter2D(image_t[:, :imshape[1] - i, :], -1, kernel_motion_blur)
        i += imshape[1] // 25 - count
        count += 1

    image_RGB = image_t

    return image_RGB


def add_speed(img_data, speed_coeff):  # (0<=speed_coeff<=1)
    count_t = int(10 * speed_coeff)
    image_RGB = [apply_motion_blur(img, count_t) for img in img_data]

    return image_RGB
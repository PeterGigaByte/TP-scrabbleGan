import os

import PIL
import cv2
import numpy as np
from PIL import Image
from skimage import exposure

path = "C:\\Users\\barad\\PycharmProjects\\TP\\scrabble-gan\\res\\data\\iamDB\\words-Reading\\"
pathAugmentation = "C:\\Users\\barad\\PycharmProjects\\TP\\"
bucket_size = 17


def deleteNoWords():
    char_vec = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i in range(1, bucket_size + 1, 1):
        try:
            reading_dir = path + str(i) + '\\'
            file_list = os.listdir(reading_dir)
            file_list = [fi for fi in file_list if fi.endswith(".txt")]
            for file in file_list:
                with open(reading_dir + file, 'r', encoding='utf8') as f:
                    for char in f.readline():
                        if char not in char_vec:
                            f.close()
                            os.remove(reading_dir + file)
                            os.remove(os.path.splitext(reading_dir + file)[0] + '.png')
                            break
        except:
            print("Folder " + str(i) + "not exist.")


def resizeImages():
    w = 16
    h = 32
    for i in range(1, bucket_size + 1, 1):
        try:
            reading_dir = path + str(i) + '\\'
            file_list = os.listdir(reading_dir)
            file_list = [fi for fi in file_list if fi.endswith(".png")]
            for file in file_list:
                file_path = reading_dir + file
                image = Image.open(file_path)
                image = image.resize((int(w * i), int(h)), PIL.Image.ANTIALIAS)
                print(w * i, h, i, file)
                print("SAVE", file_path)
                image.save(file_path)
        except:
            print("Folder " + str(i) + "not exist.")


if __name__ == "__main__":
    deleteNoWords()
    resizeImages()


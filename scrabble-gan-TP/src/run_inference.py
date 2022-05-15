# import sys
# sys.path.extend(['/home/ubuntu/workspace/scrabble-gan'])

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main():
    latent_dim = 128
    char_vec = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.'
    path_to_saved_model = 'C:/Users/barad/PycharmProjects/TP/scrabble-gan/res/out/big_ac_gan/model/generator_6'

    # number of samples to generate
    batch_size = 1
    # sample string
    sample_string = 'world'
    words_splitted = sample_string.split(" ")
    # load trained model
    imported_model = tf.saved_model.load(path_to_saved_model)
    string_l = 0
    for word in words_splitted:
        string_l += len(word)
    # inference loop
    for idx in range(1):
        plt.figure(figsize=(string_l * 16 / 96, 32 / 96), dpi=96)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        plt.subplots(nrows=1, ncols=len(words_splitted), dpi=96, sharex=False, sharey=True)
        j = 1
        for words in words_splitted:
            fake_labels = []
            words = [words] * 1
            noise = tf.random.normal([batch_size, latent_dim])
            # encode words
            for word in words:
                fake_labels.append([char_vec.index(char) for char in word])
            fake_labels = np.array(fake_labels, np.int32)

            # run inference process
            predictions = imported_model([noise, fake_labels], training=False)
            # transform values into range [0, 1]
            predictions = (predictions + 1) / 2.0

            # plot results
            for i in range(predictions.shape[0]):
                plt.subplot(1, len(words_splitted), j)

                #axarr[j].imshow(predictions[i, :, :, 0], cmap='gray')
                plt.imshow(predictions[i, :, :, 0], cmap='gray')

                # plt.text(0, -1, "".join([char_vec[label] for label in fake_labels[i]]))
                plt.axis('off')

                j += 1
        #f.show()

        plt.show()


def customGenerate(model, word):
    latent_dim = 128
    char_vec = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    path_to_saved_model = 'C:/Users/barad/PycharmProjects/TP/scrabble-gan/res/out/big_ac_gan/model/generator_' + str(
        model)

    # number of samples to generate
    batch_size = 10
    # sample string
    sample_string = word
    # load trained model
    imported_model = tf.saved_model.load(path_to_saved_model)

    # inference loop
    for idx in range(1):
        fake_labels = []
        words = [sample_string] * 10
        noise = tf.random.normal([batch_size, latent_dim])
        # encode words
        for word in words:
            fake_labels.append([char_vec.index(char) for char in word])
        fake_labels = np.array(fake_labels, np.int32)

        # run inference process
        predictions = imported_model([noise, fake_labels], training=False)
        # transform values into range [0, 1]
        predictions = (predictions + 1) / 2.0

        # plot results
        for i in range(predictions.shape[0]):
            plt.subplot(10, 1, i + 1)
            #plt.imshow(predictions[i, :, :, 0], cmap='gray')
            plt.text(0, -1, "".join([char_vec[label] for label in fake_labels[i]]))
            plt.axis('off')
        plt.savefig('C:/Users/barad/PycharmProjects/TP/scrabble-gan/res/tests/' + str(model) + '/' + str(word) + '.png')
        plt.show()


def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames


def createOneImage(model):
    images, filenames = load_images_from_folder('C:/Users/barad/PycharmProjects/TP/scrabble-gan/res/tests/' + str(model) + '/')
    fig = plt.figure(figsize=(15, 20))
    rows = 3
    columns = 3
    for i in range(len(images) - 1):
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
        plt.title(str(filenames[i]).removesuffix(".png"))
    plt.savefig('C:/Users/barad/PycharmProjects/TP/scrabble-gan/res/tests/' + str(model) +'.png')


def allModels(models):
    words = ["machinelearning","machinelearningdsadas"]
    for i in range(models):
        if not os.path.exists('C:/Users/barad/PycharmProjects/TP/scrabble-gan/res/tests/' + str(i)):
            os.makedirs('C:/Users/barad/PycharmProjects/TP/scrabble-gan/res/tests/' + str(i))
        #for word in words:
            #customGenerate(i, word)
        createOneImage(i)


if __name__ == "__main__":
    main()
    # main()

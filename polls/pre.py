from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from tqdm import tqdm
from keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import ImageFile
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, Adamax
from keras.callbacks import ModelCheckpoint
import cv2
import matplotlib.pyplot as plt
import requests
import os

def dog(img):
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # define function to load train, test, and validation datasets
    def load_dataset(path):
        data = load_files(path)
        dog_files = np.array(data['filenames'])
        dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
        return dog_files, dog_targets

    # load list of dog names
    dog_names = [item[26:-1] for item in sorted(glob("polls/dogImages/teest/*/"))]
    '''
    # print statistics about the dataset
    print('There are %d total dog categories.' % len(dog_names))
    print('There are %d total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
    print('There are %d training dog images.' % len(train_files))
    print('There are %d validation dog images.' % len(valid_files))
    print('There are %d test dog images.'% len(test_files))
    '''
    # define ResNet50 model
    ResNet50_model = ResNet50(weights='imagenet')

    def path_to_tensor(img_path):
        # loads RGB image as PIL.Image.Image type
        img = image.load_img(img_path, target_size=(224, 224))
        # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        x = image.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        return np.expand_dims(x, axis=0)

    def paths_to_tensor(img_paths):
        list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
        return np.vstack(list_of_tensors)

    def ResNet50_predict_labels(img_path):
        # returns prediction vector for image located at img_path
        img = preprocess_input(path_to_tensor(img_path))
        return np.argmax(ResNet50_model.predict(img))

    ### returns "True" if a dog is detected in the image stored at img_path
    def dog_detector(img_path):
        prediction = ResNet50_predict_labels(img_path)
        return ((prediction <= 268) & (prediction >= 151))
    '''
    human_count = 0
    for img in human_files_short:
        isDog = dog_detector(img)
        if isDog:
            human_count += 1
        percentage = (human_count/len(human_files_short)) * 100
    print('Percentage of humans misclassified as dogs:: {}%'.format(percentage))
    
    dog_count = 0
    for img in dog_files_short:
        isDog = dog_detector(img)
        if isDog:
            dog_count += 1
        percentage = (dog_count/len(dog_files_short)) * 100
    print('Percentage of dogs correctly classified as dogs: {}%'.format(percentage))
    '''
    '''
    # pre-process the data for Keras
    train_tensors = paths_to_tensor(train_files).astype('float32')/255
    valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
    test_tensors = paths_to_tensor(test_files).astype('float32')/255
    
    # create and configure augmented image generator
    datagen = ImageDataGenerator(
        width_shift_range=0.1,  # randomly shift images horizontally (10% of total width)
        height_shift_range=0.1,  # randomly shift images vertically (10% of total height)
        horizontal_flip=True) # randomly flip images horizontally
    
    # fit augmented image generator on data
    datagen.fit(train_tensors)
    
    model = Sequential()
    model.add(BatchNormalization(input_shape=(224, 224, 3)))
    model.add(Conv2D(filters=16, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())
    
    model.add(Conv2D(filters=32, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())
    
    model.add(Conv2D(filters=64, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())
    
    model.add(Conv2D(filters=128, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())
    
    model.add(Conv2D(filters=256, kernel_size=3, kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(BatchNormalization())
    
    model.add(GlobalAveragePooling2D())
    
    model.add(Dense(133, activation='softmax'))
    
    model.summary()
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    epochs = 10
    batch_size = 20
    
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.bestaugmented.from_scratch.hdf5',
                                   verbose=1, save_best_only=True)
    
    ### Using Image Augmentation
    model.fit_generator(datagen.flow(train_tensors, train_targets, batch_size=batch_size),
                        validation_data=(valid_tensors, valid_targets),
                        steps_per_epoch=train_tensors.shape[0] // batch_size,
                        epochs=epochs, callbacks=[checkpointer], verbose=1)
    
    model.load_weights('saved_models/weights.bestaugmented.from_scratch.hdf5')
    
    # get index of predicted dog breed for each image in test set
    dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
    
    # report test accuracy
    test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)
    
    batch_size = 20
    epochs = 5
    
    model.fit_generator(datagen.flow(train_tensors, train_targets, batch_size=batch_size),
                        validation_data=(valid_tensors, valid_targets),
                        steps_per_epoch=train_tensors.shape[0] // batch_size,
                        epochs=epochs, callbacks=[checkpointer], verbose=1)
    
    # get index of predicted dog breed for each image in test set
    dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
    
    # report test accuracy
    test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)
    '''

    ### TODO: Obtain bottleneck features from another pre-trained CNN.
    bottleneck_features = np.load('polls/bottleneck_features/DogResnet50Data.npz')
    train_ResNet50 = bottleneck_features['train']
    #valid_ResNet50 = bottleneck_features['valid']
    #test_ResNet50 = bottleneck_features['test']

    ResNet_model = Sequential()
    ResNet_model.add(GlobalAveragePooling2D(input_shape=train_ResNet50.shape[1:]))
    ResNet_model.add(Dense(133, activation='softmax'))

    ResNet_model.summary()

    ResNet_model.compile(loss='categorical_crossentropy', optimizer=Adamax(lr=0.002), metrics=['accuracy'])
    '''
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best_adamax.ResNet50.hdf5',
                                   verbose=1, save_best_only=True)
    
    epochs = 25
    batch_size = 64
    
    ResNet_model.fit(train_ResNet50, train_targets,
              validation_data=(valid_ResNet50, valid_targets),
              epochs=epochs, batch_size=batch_size, callbacks=[checkpointer], verbose=1)
    
    opt = Adamax(lr=0.0002)
    '''
    ### Load the model weights with the best validation loss.
    ResNet_model.load_weights('polls/saved_models/weights.best_adamax.ResNet50.hdf5')

    ### TODO: Calculate classification accuracy on the test dataset.
    '''
    # get index of predicted dog breed for each image in test set
    ResNet50_predictions = [np.argmax(ResNet_model.predict(np.expand_dims(feature, axis=0))) for feature in test_ResNet50]

    # report test accuracy
    test_accuracy = 100*np.sum(np.array(ResNet50_predictions)==np.argmax(test_targets, axis=1))/len(ResNet50_predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)
    '''
    def extract_Resnet50(tensor):
        from keras.applications.resnet50 import ResNet50, preprocess_input
        return ResNet50(weights='imagenet', include_top=False, pooling="avg").predict(preprocess_input(tensor))

    def ResNet50_predict_breed(img_path):
        # extract bottleneck features
        bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
        # feature changebv
        bottleneck_feature = np.expand_dims(bottleneck_feature, axis=0)
        bottleneck_feature = np.expand_dims(bottleneck_feature, axis=0)
        #bottleneck_feature = np.expand_dims(bottleneck_feature, axis=0)
        # obtain predicted vector
        predicted_vector = ResNet_model.predict(bottleneck_feature)
        # return dog breed that is predicted by the model
        breed = dog_names[np.argmax(predicted_vector)]
        #img = cv2.imread(img_path)
        #cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #imgplot = plt.imshow(cv_rgb)
        if dog_detector(img_path) == True:
            return format(breed)
        else:
            return print("If this person were a dog, the breed would be a {}".format(breed))

    def dog_detector(img_path):
        prediction = ResNet50_predict_labels(img_path)
        return ((prediction <= 268) & (prediction >= 151))

    def predict_breed(img_path):
        breed = ResNet50_predict_breed(img_path)
        return breed

    return predict_breed(img)
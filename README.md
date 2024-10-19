# WildGuard-AI
Our project, "WildGuard AI," aims to mitigate human-wildlife conflicts by using deep learning to detect wild animals straying into villages in real-time.


Wild animals straying into villages can be a significant concern, particularly in areas near forests or wildlife habitats. This phenomenon often occurs due to habitat destruction, urban expansion, food scarcity, or seasonal migrations. When wild animals, such as elephants, leopards, or even smaller animals like monkeys, enter human settlements, it can lead to conflicts that may endanger both animals and people.

The system uses CCTV cameras or drones equipped with AI models trained to detect various wild animals, such as elephants and leopards, near human settlements. The deep learning model is trained on large wildlife datasets and fine-tuned for the specific regions and species common in the area. Upon detecting an animal, the system triggers alerts to local authorities or residents, enabling timely intervention.


# DATASET

https://www.kaggle.com/datasets/chandrug/wildanimaldataset

# CODE

```python
import os
import zipfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import random
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout



local_zip = "animal.zip"
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('animal')
zip_ref.close()

DATA_DIR = 'animal'

# Subdirectories for each class
data_dir_bear = os.path.join(DATA_DIR, 'animal/bear_png')
data_dir_chinkara = os.path.join(DATA_DIR, 'animal/chinkara')


# os.listdir returns a list containing all files under the given dir
print(f"There are {len(os.listdir(data_dir_bear))} images of bear.")
print(f"There are {len(os.listdir(data_dir_chinkara))} images of chinkara.")

bear_filenames = [os.path.join(data_dir_bear, filename) for filename in os.listdir(data_dir_bear)]
chinkara_filenames = [os.path.join(data_dir_chinkara, filename) for filename in os.listdir(data_dir_chinkara)]
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle('bear and chinkara', fontsize=16)

# Plot the first 4 images of each class
for i, bear_image in enumerate(bear_filenames[:4]):
    img = tf.keras.utils.load_img(bear_image)
    axes[0, i].imshow(img)
    axes[0, i].set_title(f'Example bear {i}')

for i, chinkara_image in enumerate(chinkara_filenames[:4]):
    img = tf.keras.utils.load_img(chinkara_image)
    axes[1, i].imshow(img)
    axes[1, i].set_title(f'Example chinkara {i}')

plt.show()

main_folder = 'animal/animal'  # Path to the subfolder containing 7 subdirectories
train_folder = 'animal/train'
test_folder = 'animal/test'

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

train_ratio = 0.8  
test_ratio = 0.2

# Allowed image extensions (add more if needed)
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# Loop through each subfolder (each class)
for class_folder in os.listdir(main_folder):
    class_path = os.path.join(main_folder, class_folder)
    
    # Check if it's a directory (class folder)
    if os.path.isdir(class_path):
        # Get list of all image files in the subfolder
        images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f)) and os.path.splitext(f)[1].lower() in image_extensions]
        
        # Skip the folder if no images are found
        if len(images) == 0:
            print(f"No images found in folder: {class_folder}. Skipping...")
            continue
        
        # Split the images into train and test
        train_images, test_images = train_test_split(images, test_size=test_ratio, random_state=42)
        
        # Create corresponding class directories in train and test folders
        os.makedirs(os.path.join(train_folder, class_folder), exist_ok=True)
        os.makedirs(os.path.join(test_folder, class_folder), exist_ok=True)
        
        # Move the train images
        for image in train_images:
            src = os.path.join(class_path, image)
            dest = os.path.join(train_folder, class_folder, image)
            shutil.copy(src, dest)
        
        # Move the test images
        for image in test_images:
            src = os.path.join(class_path, image)
            dest = os.path.join(test_folder, class_folder, image)
            shutil.copy(src, dest)

print("Image splitting into train and test sets is completed!")

import os

# Set the paths
train_folder = 'animal/train'
test_folder = 'animal/test'

# Function to count images in a folder
def count_images(folder):
    total_images = 0
    for class_folder in os.listdir(folder):
        class_path = os.path.join(folder, class_folder)
        if os.path.isdir(class_path):
            # Count the number of image files in the class folder
            image_count = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
            total_images += image_count
            print(f"Number of images in {class_folder}: {image_count}")
    return total_images

# Count images in train and test folders
train_count = count_images(train_folder)
test_count = count_images(test_folder)

print(f"\nTotal images in train folder: {train_count}")
print(f"Total images in test folder: {test_count}")

train_folder = 'animal/train'
test_folder = 'animal/test'

# Image parameters
image_size = (128, 128)  # Resize all images to 128x128
batch_size = 32  # Number of images to process in a batch
num_classes = len(os.listdir(train_folder))  # Number of classes

# Data augmentation and loading images from directory
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalize pixel values
    shear_range=0.2,          # Randomly apply shearing
    zoom_range=0.2,           # Randomly zoom in images
    horizontal_flip=True,     # Randomly flip images horizontally
)

test_datagen = ImageDataGenerator(rescale=1./255)  # Only normalize test images

# Load train and test datasets
train_generator = train_datagen.flow_from_directory(
    train_folder,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'  # For multi-class classification
)

test_generator = test_datagen.flow_from_directory(
    test_folder,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Define the CNN model
model = Sequential()

# Convolution + Pooling layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output from convolution layers and connect to fully connected layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dropout to reduce overfitting
model.add(Dense(num_classes, activation='softmax'))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=10,  # You can increase this value for better training
    validation_data=test_generator
)

# Plot training history (accuracy and loss)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(10)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc * 100:.2f}%")


```

# OUTPUT:

![image](https://github.com/user-attachments/assets/53d1f95b-b418-4f37-9f71-b2fa6fdec136)

![image](https://github.com/user-attachments/assets/ae52c6c4-c404-4ef6-991e-77ca01a2f3b7)


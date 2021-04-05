# BLOCK #1
# this block loads a dataset from tensorflow repository
# and also it include tensorflow lib
import tensorflow_datasets as tfds
import tensorflow as tf

print(tf.__version__)

# BLOCK #2
# now it loads data for training and data for testing, also it loads info about
# this dataset
train_data, info = tfds.load("fashion_mnist", with_info=True, split="train")
test_data = tfds.load("fashion_mnist", split="test")

# BLOCK #2.1
# show an information about loaded dataset
# print(info)

# BLOCK #3
# give names of each in the dataset
names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


# BLOCK #4
# import different stuff for plotting
# from matplotlib import rcParams
# from matplotlib import pyplot as plt
#
# rcParams["figure.figsize"] = [10, 10]
# rcParams['xtick.labelbottom'] = False

# BLOCK #5
# finaly plot it on the
# for idx, elem in enumerate(train_data.take(25)):
#     plt.subplot(5, 5, idx + 1, title=names[elem['label'].numpy()])
#     plt.imshow(elem['image'][:, :, 0])

# windows code
# !!! WARNING it used in my win-system.
# plt.show()


# BLOCK #6
# this function does the simplest pre-processing, it just unzip loaded data
# in more comfortable way
def preprocessing(data):
    x = tf.reshape(data["image"], [-1])
    y = data["label"]
    return x, y


# BLOCK #7
def preprocessing_with_norm(data):
    x = tf.reshape(data["image"], [-1]) / 255
    y = data["label"]
    return x, y


# BLOCK #8
import tensorflow_addons as tfa
from random import seed
from random import randint
from random import random

seed(1)


def do_augmentation(image):
    rand_augment_code = randint(0, 28)
    angle = 0.2 * random()
    shear = 0.15 * random()
    if rand_augment_code < 4:
        x = tf.image.random_flip_left_right(image)
    elif rand_augment_code < 8:
        x = tfa.image.rotate(image, angle, fill_mode="constant", fill_value=0)
    elif rand_augment_code < 12:
        x = tfa.image.rotate(image, -angle, fill_mode="constant", fill_value=0)
    elif rand_augment_code < 16:
        x = tfa.image.shear_x(image, shear)
    elif rand_augment_code < 20:
        x = tfa.image.shear_y(image, -shear)
    elif rand_augment_code < 24:
        x = tfa.image.shear_y(image, shear)
    else:
        x = tfa.image.shear_y(image, -shear)
    return x


def preprocessing_with_augmentaion(data):
    x = do_augmentation(data["image"])
    x = tf.reshape(x, [-1])
    y = data["label"]
    return x, y


def preprocessing_with_aug_norm(data):
    x = do_augmentation(data["image"])
    x = tf.reshape(x, [-1]) / 255
    y = data["label"]
    return x, y


# BLOCK #8
# pre-process the train and the test data
train_data_pre_1 = train_data.map(preprocessing_with_aug_norm)
train_data_pre_2 = train_data.map(preprocessing_with_norm)
train_data_pre = train_data_pre_1.concatenate(train_data_pre_2)

test_data_pre = test_data.map(preprocessing_with_norm)

batch_size = 64
train_data_pre = train_data_pre.batch(batch_size)
test_data_pre = test_data_pre.batch(batch_size)


# BLOCK #9
# this function defines base model (!!!WARNING, in the task I can't modify it).
# then it creates model and print model summary
# Cause we have the input 28*28*1 (gray_scale) = 784 - here is an input layer
def base_model():
    inputs = tf.keras.Input(shape=(784,))
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


model = base_model()
# model.summary()

# BLOCK #10
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

@tf.function
def regularization_l1(trainable_variables, var_lambda=0.01):
    reg = 0.0
    for x in trainable_variables:
        reg = reg + tf.math.reduce_sum(tf.math.abs(x))
    return var_lambda * reg

@tf.function
def regularization_l2(trainable_variables, var_lambda=0.01):
    reg = 0.0
    for x in trainable_variables:
        reg = reg + tf.math.reduce_sum(tf.math.square(x))
    reg = tf.math.sqrt(reg)
    return var_lambda * reg

@tf.function
def regularization_l1_l2(trainable_variables, var_lambda_l1=0.01, var_lambda_l2=0.01):
    reg_l1 = regularization_l1(trainable_variables, var_lambda=var_lambda_l1)
    reg_l2 = regularization_l2(trainable_variables, var_lambda=var_lambda_l2)
    return reg_l1 + reg_l2



@tf.function
def train_step(x, y, lr=0.0005):
    with tf.GradientTape() as tape:
        pred = model(x)
        loss = loss_object(y, pred)
        regl = regularization_l2(model.trainable_variables)
        gradients = tape.gradient(loss + regl, model.trainable_variables)

        new_weights = []
        for w, grad in zip(model.trainable_variables, gradients):
            new_weights.append(w - lr * grad)

        for v, w in zip(model.trainable_variables, new_weights):
            v.assign(w)
    pred_nums = tf.math.argmax(pred, axis=1)
    equality = tf.math.equal(pred_nums, y)
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))

    return loss, accuracy


# BLOC #11
import numpy as np

x_test = np.concatenate([x for x, y in test_data_pre], axis=0)
y_test = np.concatenate([y for x, y in test_data_pre], axis=0)


# BLOCK #12
def calc_acc():
    pred = model.predict(x_test)
    acc = np.sum(np.argmax(pred, axis=-1) == y_test) / len(y_test)
    return acc


# BLOCK #13
print("Show initial accuracity")
print("Test acc: ", calc_acc())

# BLOCK #14
def schedule_lr(epoch_count, epoch):
    if epoch < epoch_count / 5:
        return 0.001
    elif epoch < 2 * epoch_count / 5:
        return 0.0009
    elif epoch < 3 * epoch_count / 5:
        return 0.0007
    elif epoch < 4 * epoch_count / 5:
        return 0.0008
    else:
        return 0.0005

# BLOCK #14
from tqdm import tqdm

total_loss_acc_array = []

epoch_count = 50
count_batches = len(train_data_pre)
for epoch in range(epoch_count):
    lr = schedule_lr(epoch_count, epoch)
    total_loss = 0.0
    total_accuracy = 0.0
    print("\nEpoch {}:".format(str(epoch + 1) + "/" + str(epoch_count)))
    for step, (x, y) in enumerate(tqdm(train_data_pre)):
        training_result = train_step(x, y, lr=lr)
        total_loss = total_loss + training_result[0].numpy()
        total_accuracy = total_accuracy + training_result[1].numpy()
    total_loss = total_loss / count_batches
    total_accuracy = total_accuracy / count_batches
    total_loss_acc_array.append([epoch + 1, total_loss, total_accuracy])
    print("loss =", total_loss, " acc =", total_accuracy)

# BLOCK #15
import matplotlib.pyplot as plt

epoch_array = []
loss_array = []
acc_array = []
for inx in range(len(total_loss_acc_array)):
    epoch_array.append(total_loss_acc_array[inx][0])
    loss_array.append(total_loss_acc_array[inx][1])
    acc_array.append(total_loss_acc_array[inx][2])

max_loss = max(loss_array)
loss_array = loss_array / max_loss

plt.plot(epoch_array, loss_array, label="loss")
plt.plot(epoch_array, acc_array, label="acc")
plt.xlabel("epochs")
plt.title("total loss and accuracy")

plt.legend()
plt.show()

# BLOCK #16
print("\n\nShow last accuracity")
print("Test acc: ", calc_acc())

# BLOCK #17
# from matplotlib import rcParams
# from matplotlib import pyplot as plt
#
# rcParams["figure.figsize"] = [10, 10]
# rcParams['xtick.labelbottom'] = False

# BLOCK #18
# finaly plot it on the
# test_pred = model.predict(x_test)
#
# for idx, elem in enumerate(test_data.take(25)):
#     pred_idx = np.argmax(test_pred[idx])
#     true_idx = y_test[idx]
#     plt.subplot(5, 5, idx + 1, title=(names[pred_idx] + "(" + names[true_idx] + ")"))
#     plt.imshow(elem['image'][:, :, 0])

# windows code
# !!! WARNING it used in my win-system.
# plt.show()

# BLOCK 19
from matplotlib import pyplot

test_pred = model.predict(x_test)
fig = pyplot.figure(figsize=(20, 8))

for idx, elem in enumerate(test_data.take(32)):
    ax = fig.add_subplot(4, 8, idx + 1, xticks=[], yticks=[])
    ax.imshow(elem['image'][:, :, 0])
    pred_idx = np.argmax(test_pred[idx])
    true_idx = y_test[idx]
    ax.set_title("{} ({})".format(names[pred_idx], names[true_idx]),
                 color=("green" if pred_idx == true_idx else "red"))
pyplot.show()

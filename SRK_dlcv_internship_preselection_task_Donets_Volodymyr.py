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
# pre-process the train and the test data
train_data_pre = train_data.map(preprocessing)
test_data_pre = test_data.map(preprocessing)

batch_size = 64
train_data_pre = train_data_pre.batch(batch_size)
test_data_pre = test_data_pre.batch(batch_size)


# BLOCK #8
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
def train_step(x, y):
    lr = 0.0004
    with tf.GradientTape() as tape:
        pred = model(x)
        loss = loss_object(y, pred)
        gradients = tape.gradient(loss, model.trainable_variables)

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
from tqdm import tqdm

epoch_num = 30
count_batches = len(train_data_pre)
for epoch in range(epoch_num):
    print("\nEpoch {}:".format(str(epoch + 1) + "/" + str(epoch_num)))
    total_loss = 0.0
    total_accuracy = 0.0
    for step, (x, y) in enumerate(tqdm(train_data_pre)):
        training_result = train_step(x, y)
        total_loss = total_loss + training_result[0].numpy()
        total_accuracy = total_accuracy + training_result[1].numpy()
    total_loss = total_loss / count_batches
    total_accuracy = total_accuracy / count_batches
    print("loss =", total_loss, " acc =", total_accuracy)

# BLOCK #15
print("\n\nShow last accuracity")
print("Test acc: ", calc_acc())

# BLOCK #16
# from matplotlib import rcParams
# from matplotlib import pyplot as plt
#
# rcParams["figure.figsize"] = [10, 10]
# rcParams['xtick.labelbottom'] = False

# BLOCK #17
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

# BLOCK 18
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
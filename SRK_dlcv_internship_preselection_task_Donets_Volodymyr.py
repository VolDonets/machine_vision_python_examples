# BLOCK #1
import tensorflow_datasets as tfds
import tensorflow as tf
print(tf.__version__)

# BLOCK #2
train_data, info  = tfds.load("fashion_mnist", with_info = True, split = "train")
test_data = tfds.load("fashion_mnist", split = "test")

# BLOCK #3
names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker",	"Bag",	"Ankle boot"]

# BLOCK #4
from matplotlib import rcParams
from matplotlib import pyplot as plt
rcParams["figure.figsize"] = [10,10]
rcParams['xtick.labelbottom'] = False

# BLOCK #5
for idx, elem in enumerate(train_data.take(25)):
  plt.subplot(5, 5, idx+1, title = names[elem['label'].numpy()] )
  plt.imshow(elem['image'][:,:,0])

# windows code
plt.show()

# BLOCK #6
def preprocessing(data):
    x = tf.reshape(data["image"], [-1])
    y = data["label"]
    return x, y

# BLOCK #7
train_data_pre = train_data.map(preprocessing)
test_data_pre = test_data.map(preprocessing)

batch_size = 64
train_data_pre = train_data_pre.batch(batch_size)
test_data_pre = test_data_pre.batch(batch_size)

# BLOCK #8
def base_model():
  inputs = tf.keras.Input(shape=(784,))
  x = tf.keras.layers.Dense(64, activation='relu')(inputs)
  x = tf.keras.layers.Dense(64, activation='relu')(x)
  outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  return model

model = base_model()
model.summary()

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
      new_weights.append(w - lr*grad)

    for v, w in zip (model.trainable_variables, new_weights):
      v.assign(w)

# BLOCK #11
from tqdm import tqdm
epoch_num = 30
for epoch in range(epoch_num):
  print("epoch {}:".format(epoch))
  for step, (x,y) in enumerate(tqdm(train_data_pre) ):
    train_step(x,y)
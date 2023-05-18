
import os
import sys
import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers


tf.random.set_seed(123)

#Inspired from U-Net with our own additions
#Extracting_Dataset
folder = "/dataset/"
if not os.path.exists(os.path.abspath(".") + folder):
    zip = tf.keras.utils.get_file(
        "val.tar.gz",
        cache_subdir=os.path.abspath("."),
        origin="http://diode-dataset.s3.amazonaws.com/val.tar.gz",
        extract=True,
    )



path = "val/indoors"

imagelist = []

for root, dirs, files in os.walk(path):
    for file in files:
        imagelist.append(os.path.join(root, file))

imagelist.sort()
data = {
    "image": [x for x in imagelist if x.endswith(".png")],
    "depth": [x for x in imagelist if x.endswith("_depth.npy")],
    "mask": [x for x in imagelist if x.endswith("_depth_mask.npy")],
}

df = pd.DataFrame(data)

df = df.sample(frac=1, random_state=42)



H = 256
W = 256
Learn_rate = 0.0002
epochs = 30
Batch_size = 32



class Generate_data(tf.keras.utils.Sequence):
    def __init__(self, data, Batch_size=6, dimension=(768, 1024), channels=3, shuffle=True):
        """
        Initialization
        """
        self.data = data
        self.indices = self.data.index.tolist()
        self.dim = dimension
        self.channels = channels
        self.Batch_size = Batch_size
        self.shuffle = shuffle
        self.min_depth = 0.1
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.Batch_size))

    def __getitem__(self, index):
        if (index + 1) * self.Batch_size > len(self.indices):
            self.Batch_size = len(self.indices) - index * self.Batch_size
        index = self.indices[index * self.Batch_size : (index + 1) * self.Batch_size]
        batch = [self.indices[k] for k in index]
        x, y = self.generation_data(batch)

        return x, y

    def on_epoch_end(self):

        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def load_data(self, image_path, depth_map, mask):


        image_ = cv.imread(image_path)
        image_ = cv.cvtColor(image_, cv.COLOR_BGR2RGB)
        image_ = cv.resize(image_, self.dim)
        image_ = tf.image.convert_image_dtype(image_, tf.float32)

        depth_map = np.load(depth_map).squeeze()

        mask = np.load(mask)
        mask = mask > 0

        max_depth = min(300, np.percentile(depth_map, 99))
        depth_map = np.clip(depth_map, self.min_depth, max_depth)
        depth_map = np.log(depth_map, where=mask)

        depth_map = np.ma.masked_where(~mask, depth_map)

        depth_map = np.clip(depth_map, 0.1, np.log(max_depth))
        depth_map = cv.resize(depth_map, self.dim)
        depth_map = np.expand_dims(depth_map, axis=2)
        depth_map = tf.image.convert_image_dtype(depth_map, tf.float32)

        return image_, depth_map

    def generation_data(self, batch):

        x = np.empty((self.Batch_size, *self.dim, self.channels))
        y = np.empty((self.Batch_size, *self.dim, 1))

        for i, batch_id in enumerate(batch):
            x[i,], y[i,] = self.load_data(
                self.data["image"][batch_id],
                self.data["depth"][batch_id],
                self.data["mask"][batch_id],
            )

        return x, y





class Encoder(layers.Layer):
    def __init__(
        self, filters, kernel_size=(3, 3), padding="same", strides=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.convolutionA = layers.Conv2D(filters, kernel_size, strides, padding, kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4))
        self.convolutionB = layers.Conv2D(filters, kernel_size, strides, padding, kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4))
        self.LeakyReLUA = layers.LeakyReLU(alpha=0.2)
        self.LeakyReLUB = layers.LeakyReLU(alpha=0.2)
        self.batch_na = tf.keras.layers.BatchNormalization()
        self.batch_nb = tf.keras.layers.BatchNormalization()

        self.pool = layers.MaxPool2D((2, 2), (2, 2))

    def call(self, input_tensor):
        d = self.convolutionA(input_tensor)
        x = self.batch_na(d)
        x = self.LeakyReLUA(x)

        x = self.convolutionB(x)
        x = self.batch_nb(x)
        x = self.LeakyReLUB(x)

        x += d
        p = self.pool(x)
        return x, p


class Decoder(layers.Layer):
    def __init__(
        self, filters, kernel_size=(3, 3), padding="same", strides=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.us = layers.UpSampling2D((2, 2))
        self.convolutionA = layers.Conv2D(filters, kernel_size, strides, padding, kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4),  kernel_regularizer=regularizers.l2(0.01))
        self.convolutionB = layers.Conv2D(filters, kernel_size, strides, padding, kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4),  kernel_regularizer=regularizers.l2(0.01))
        self.LeakyReLUA = layers.LeakyReLU(alpha=0.2)
        self.LeakyReLUB = layers.LeakyReLU(alpha=0.2)
        self.batch_na = tf.keras.layers.BatchNormalization()
        self.batch_nb = tf.keras.layers.BatchNormalization()
        self.conc = layers.Concatenate()

    def call(self, x, skip):
        x = self.us(x)
        concat = self.conc([x, skip])
        x = self.convolutionA(concat)
        x = self.batch_na(x)
        x = self.LeakyReLUA(x)

        x = self.convolutionB(x)
        x = self.batch_nb(x)
        x = self.LeakyReLUB(x)

        return x


class BottleNeckBlock(layers.Layer):
    def __init__(
        self, filters, kernel_size=(3, 3), padding="same", strides=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.convolutionA = layers.Conv2D(filters, kernel_size, strides, padding, kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4), kernel_regularizer=regularizers.l2(0.01))
        self.convolutionB = layers.Conv2D(filters, kernel_size, strides, padding, kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4),  kernel_regularizer=regularizers.l2(0.01))
        self.LeakyReLUA = layers.LeakyReLU(alpha=0.2)
        self.LeakyReLUB = layers.LeakyReLU(alpha=0.2)

    def call(self, x):
        x = self.convolutionA(x)
        x = self.LeakyReLUA(x)
        x = self.convolutionB(x)
        x = self.LeakyReLUB(x)
        return x


class Depth_Estimation_Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.ssim_loss_weight = 0.85
        self.l1_loss_weight = 0.1
        self.edge_loss_weight = 0.9
        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        f = [16, 32, 64, 128, 256]
        self.encoder_blocks = [
            Encoder(f[0]),
            Encoder(f[1]),
            Encoder(f[2]),
            Encoder(f[3]),
        ]
        self.bottle_neck_block = BottleNeckBlock(f[4])
        self.decoder_blocks = [
            Decoder(f[3]),
            Decoder(f[2]),
            Decoder(f[1]),
            Decoder(f[0]),
        ]
        self.convolution_layer = layers.Conv2D(1, (1, 1), padding="valid", activation="sigmoid", kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4))

    def calculate_loss(self, target, pred):

        dy_true, dx_true = tf.image.image_gradients(target)
        dy_pred, dx_pred = tf.image.image_gradients(pred)
        weights_x = tf.exp(tf.reduce_mean(tf.abs(dx_true)))
        weights_y = tf.exp(tf.reduce_mean(tf.abs(dy_true)))

 
        smoothness_x = dx_pred * weights_x
        smoothness_y = dy_pred * weights_y

        depth_smoothness_loss = tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(
            abs(smoothness_y)
        )


        ssim_loss = tf.reduce_mean(
            1
            - tf.image.ssim(
                target, pred, max_val=W, filter_size=7, k1=0.01 ** 2, k2=0.03 ** 2
            )
        )

        l1_loss = tf.reduce_mean(tf.abs(target - pred))

        loss = (
            (self.ssim_loss_weight * ssim_loss)
            + (self.l1_loss_weight * l1_loss)
            + (self.edge_loss_weight * depth_smoothness_loss)
        )

        return loss

    @property
    def metrics(self):
        return [self.loss_metric]

    def train_step(self, batch_data):
        input, target = batch_data
        with tf.GradientTape() as tape:
            pred = self(input, training=True)
            loss = self.calculate_loss(target, pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_metric.update_state(loss)
        return {
            "loss": self.loss_metric.result(),
        }

    def test_step(self, batch_data):
        input, target = batch_data

        pred = self(input, training=False)
        loss = self.calculate_loss(target, pred)

        self.loss_metric.update_state(loss)
        return {
            "loss": self.loss_metric.result(),
        }

    def call(self, x):
        c1, p1 = self.encoder_blocks[0](x)
        c2, p2 = self.encoder_blocks[1](p1)
        c3, p3 = self.encoder_blocks[2](p2)
        c4, p4 = self.encoder_blocks[3](p3)

        bn = self.bottle_neck_block(p4)

        u1 = self.decoder_blocks[0](bn, c4)
        u2 = self.decoder_blocks[1](u1, c3)
        u3 = self.decoder_blocks[2](u2, c2)
        u4 = self.decoder_blocks[3](u3, c1)

        return self.convolution_layer(u4)

def visualize_depth_map(samples, test=False, model=None):
    input, target = samples
    cmap = plt.cm.jet
    cmap.set_bad(color="black")

    if test:
        pred = model.predict(input)
        fig, ax = plt.subplots(6, 3, figsize=(50, 50))
        for i in range(6):
            ax[i, 0].imshow((input[i].squeeze()))
            ax[i, 1].imshow((target[i].squeeze()), cmap=cmap)
            ax[i, 2].imshow((pred[i].squeeze()), cmap=cmap)

    else:
        fig, ax = plt.subplots(6, 2, figsize=(50, 50))
        for i in range(6):
            ax[i, 0].imshow((input[i].squeeze()))
            ax[i, 1].imshow((target[i].squeeze()), cmap=cmap)

class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        if tf.math.is_nan(logs.get('loss')):
            print("Got Loss - Nan")
            self.model.stop_training = True


visualize_samples = next(
    iter(Generate_data(data=df, Batch_size=6, dim=(H, W)))
)
visualize_depth_map(visualize_samples)

LR = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.002,
            decay_steps=100000,
            decay_rate=0.90)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LR,
    amsgrad=False, clipnorm=1.0
)

model = Depth_Estimation_Model()

cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)

model.compile(optimizer, loss=cross_entropy)

train_loader = Generate_data(data=df[:260].reset_index(drop="true"), Batch_size=Batch_size, dim=(H, W))
validation_loader = Generate_data(data=df[260:].reset_index(drop="true"), Batch_size=Batch_size, dim=(H, W))

reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=3)
early_callback = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=3)


history = model.fit(
    train_loader,
    epochs=epochs,
    validation_data=validation_loader,
    callbacks=[reduce_lr_callback, early_callback, LossHistory()])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


test_loader = next(iter( Generate_data(data=df[265:].reset_index(drop="true"), Batch_size=6, dim=(H, W) )))

visualize_depth_map(test_loader, test=True, model=model)

test_loader = next(iter(Generate_data(data=df[300:].reset_index(drop="true"), Batch_size=6, dim=(H, W))))

visualize_depth_map(test_loader, test=True, model=model)

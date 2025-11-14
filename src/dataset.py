# dataset.py
"""
Beginner-friendly TensorFlow dataset loader with tf.data pipeline, augmentation, batching, prefetching, shuffling.
"""
import tensorflow as tf
import numpy as np
import os

def get_dataset(data_dir, batch_size=32, img_size=(600, 600), augment=True):
    # Get file paths and labels
    cad_paths = tf.io.gfile.glob(os.path.join(data_dir, 'CAD', '*.npy'))
    noncad_paths = tf.io.gfile.glob(os.path.join(data_dir, 'NonCAD', '*.npy'))
    paths = cad_paths + noncad_paths
    labels = [1]*len(cad_paths) + [0]*len(noncad_paths)
    files_labels = list(zip(paths, labels))
    np.random.shuffle(files_labels)
    paths, labels = zip(*files_labels)
    paths = np.array(paths)
    labels = np.array(labels)

    def preprocess(path, label):
        img = tf.numpy_function(lambda p: np.load(p.decode()), [path], tf.float32)
        img.set_shape([600, 600, 3])  # RGB shape
        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.rot90(img, k=np.random.randint(1, 4))
            img = tf.image.random_brightness(img, max_delta=0.1)
            img = tf.image.random_zoom(img, [0.95, 1.05])
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(buffer_size=len(paths))
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# Example usage
if __name__ == "__main__":
    train_ds = get_dataset('../data/train', batch_size=32)
    for images, labels in train_ds.take(1):
        print("Batch shape:", images.shape, labels.shape)

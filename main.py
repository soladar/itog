import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def load_cifar_data():
    print("Loading CIFAR-100 data...")

    data_pre_path = 'C:/Users/sheyn/Python notebook/data/last/cifar-100-python/'
    data_train_path = data_pre_path + 'train'
    data_test_path = data_pre_path + 'test'
    data_train_dict = unpickle(data_train_path)
    data_test_dict = unpickle(data_test_path)

    X_train_20 = data_train_dict[b'data']
    y_train_20 = np.array(data_train_dict[b'coarse_labels'])
    X_test_20 = data_test_dict[b'data']
    y_test_20 = np.array(data_test_dict[b'coarse_labels'])
    (X_train_100, y_train_100), (X_test_100, y_test_100) = tf.keras.datasets.cifar100.load_data()

    y_train_20 = tf.one_hot(y_train_20, depth=y_train_20.max() + 1, dtype=tf.float64)
    y_test_20 = tf.one_hot(y_test_20, depth=y_test_20.max() + 1, dtype=tf.float64)

    y_train_20 = tf.squeeze(y_train_20)
    y_test_20 = tf.squeeze(y_test_20)

    X_train_20 = X_train_20.reshape(-1, 32, 32, 3)
    X_test_20 = X_test_20.reshape(-1, 32, 32, 3)

    y_train_100 = tf.one_hot(y_train_100, depth=y_train_100.max() + 1, dtype=tf.float64)
    y_test_100 = tf.one_hot(y_test_100, depth=y_test_100.max() + 1, dtype=tf.float64)

    y_train_100 = tf.squeeze(y_train_100)
    y_test_100 = tf.squeeze(y_test_100)

    X_train_100 = X_train_100.reshape(-1, 32, 32, 3)
    X_train_100 = X_train_100.reshape(-1, 32, 32, 3)


    print("CIFAR-100 data loaded.")

    return X_train_20, y_train_20, X_test_20, y_test_20, X_train_100, y_train_100, X_test_100, y_test_100


def create_model(num_classes):
    print(f"Creating model for {num_classes} classes...")

    model = tf.keras.models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu',
                      input_shape=(32, 32, 3), padding='same'),
        layers.Conv2D(32, (3, 3),
                      activation='relu',
                      padding='same'),
        layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same'),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization()
    ])

    # Отдельный выходной слой для каждой модели с учетом количества классов
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')
    model.add(output_layer)

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['AUC', 'accuracy']
    )

    print("Model created.")

    return model


def train_model(model, X_train, y_train, X_test, y_test, num_classes):
    print(f"Training model for {num_classes} classes...")

    model.fit(X_train, y_train,
              epochs=6,
              batch_size=64,
              verbose=1,
              validation_data=(X_test, y_test))

    print(f"Model trained for {num_classes} classes.")

    return model


def save_model(model, filename):
    model.save(filename)
    print(f"Model saved to {filename}")


def main():
    X_train_20, y_train_20, X_test_20, y_test_20, X_train_100, y_train_100, X_test_100, y_test_100 = load_cifar_data()

    model_20 = create_model(num_classes=20)
    model_20 = train_model(model_20, X_train_20, y_train_20, X_test_20, y_test_20, num_classes=20)
    save_model(model_20, "model_20.h5")

    model_100 = create_model(num_classes=100)
    model_100 = train_model(model_100, X_train_100, y_train_100, X_test_100, y_test_100, num_classes=100)
    save_model(model_100, "model_100.h5")


if __name__ == "__main__":
    main()

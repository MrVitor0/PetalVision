import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os
import random

class Trainer:
    def __init__(self):
        self.model = self._load_or_build_model()

    def _build_model(self):
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(4,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(3, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model
    
    def _load_or_build_model(self):
        if os.path.exists("saved_model"):
            model = models.load_model("saved_model")
        else:
            model = self._build_model()
        return model
    

    def load_data(self):
        iris = load_iris()
        X, y = iris.data, iris.target

        # Normalize the data
        X = X / X.max()

        # Convert the target to one-hot encoding
        y = tf.keras.utils.to_categorical(y, num_classes=3)

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, y_train, X_test, y_test, X, y

    def train_model(self, X_train, y_train, epochs=50, batch_size=8, validation_split=0.1):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    def save_model(self):
        self.model.save("./saved_model")

    def evaluate_model(self, X_test, y_test, num_items_to_evaluate=None):
        if num_items_to_evaluate is None:
            num_items_to_evaluate = X_test.shape[0]

        test_indices = random.sample(range(X_test.shape[0]), num_items_to_evaluate)
        X_test_subset = X_test[test_indices]
        y_test_subset = y_test[test_indices]

        test_loss, test_accuracy = self.model.evaluate(X_test_subset, y_test_subset)
        print(f"Test accuracy: {test_accuracy}")

    def predict(self, new_sample):
        prediction = self.model.predict(tf.expand_dims(new_sample, axis=0))
        predicted_class = tf.argmax(prediction, axis=1).numpy()[0]
        return predicted_class


 
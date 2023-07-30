import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import time
from tensorflow.python.client import device_lib 

class Trainer:
    def __init__(self):
        self.model = self._build_model()

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
    
    def load_data(self):
        iris = load_iris()
        X, y = iris.data, iris.target

        # Normalize the data
        X = X / X.max()

        # Convert the target to one-hot encoding
        y = tf.keras.utils.to_categorical(y, num_classes=3)

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, y_train, X_test, y_test

    def train_model(self, X_train, y_train, epochs=50, batch_size=8, validation_split=0.1):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    def evaluate_model(self, X_test, y_test):
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test accuracy: {test_accuracy}")

    def predict(self, new_sample):
        prediction = self.model.predict(tf.expand_dims(new_sample, axis=0))
        predicted_class = tf.argmax(prediction, axis=1).numpy()[0]
        return predicted_class


    def check_device(self):
        return device_lib.list_local_devices()
    
    def gpu_benchmark(self, matrix_size=1000):
        print("Running GPU benchmark...")

        # Verificar se a GPU está disponível
        if not tf.config.list_physical_devices('GPU'):
            print("GPU not found. Make sure you have installed the correct version of TensorFlow for GPU support.")
            return

        # Criar duas matrizes aleatórias de tamanho matrix_size x matrix_size
        with tf.device('/GPU:0'):  # Acesso à primeira GPU disponível (se houver mais de uma)
            matrix_a = tf.random.normal((matrix_size, matrix_size))
            matrix_b = tf.random.normal((matrix_size, matrix_size))

            # Executar a multiplicação de matrizes na GPU e medir o tempo
            start_time = time.time()
            result = tf.matmul(matrix_a, matrix_b)
            end_time = time.time()

        elapsed_time = end_time - start_time
        print(f"Matrix multiplication time on GPU (matrix size {matrix_size}): {elapsed_time:.6f} seconds")
        return elapsed_time

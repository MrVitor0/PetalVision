from tensorflow.python.client import device_lib 
import tensorflow as tf
import time 

class Benchmark:
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

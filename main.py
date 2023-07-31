from Trainer import Trainer
from Benchmark import Benchmark
import numpy as np
def main():
    trainer = Trainer()
    benchmark = Benchmark()

    print("1. Train the model")
    print("2. Test the model")
    print("3. Evaluate the model")
    print("4. Check if GPU is available")
    print("5. Benchmark GPU")
    choice = int(input("Choose one between (1, 2, 3, 4, 5): "))

    X_train, y_train, X_test, y_test, _, _ = trainer.load_data()

    if choice == 1:
        epochs = int(input("Enter the epochs count: "))
        batch_size = int(input("Enter the batch size: "))
        validation_split = float(input("Enter the validation proportion (between 0 e 1): "))

        trainer.train_model(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        trainer.save_model()
        print("Treinamento concluído.")

    if choice == 2:
        # [sepal length, sepal width, petal length, petal width]
        new_flower1 = [5.1, 3.5, 1.4, 0.2]  # "Setosa"
        new_flower2 = [6.3, 3.3, 6.0, 2.5]  # "Virginica"

        X_train = np.array(trainer.load_data()[0])
        new_flower1 = new_flower1 / X_train.max()
        new_flower2 = new_flower2 / X_train.max()

        # Fazer previsões usando o modelo treinado
        predicted_class1 = trainer.predict(new_flower1)
        predicted_class2 = trainer.predict(new_flower2)

        species_names = ["Setosa", "Versicolor", "Virginica"]
        predicted_species1 = species_names[predicted_class1]
        predicted_species2 = species_names[predicted_class2]

        print(f"Predicted species for new flower 1: {predicted_species1}")
        print(f"Predicted species for new flower 2: {predicted_species2}")

    elif choice == 3:
        # Avaliar o modelo com apenas 5 itens do conjunto de teste
        num_items_to_evaluate = 5
        trainer.evaluate_model(X_test, y_test, num_items_to_evaluate)


    elif choice == 4:
        print(benchmark.check_device())
        
    elif choice == 5:
        matrix_size = int(input("Enter the matrix size: "))
        benchmark.gpu_benchmark(matrix_size=matrix_size)

    else:
        print("Error. Choice between 1, 2, 3, 4, 5")

if __name__ == "__main__":
    main()

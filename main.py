from Trainer import Trainer

def main():
    trainer = Trainer()

    print("1. Treinar o modelo")
    print("2. Testar o modelo")
    print("3. Verificar dispositivo de processamento")
    print("4. Benchmark GPU")
    choice = int(input("Escolha uma opção (1, 2, 3 ou 4): "))

    X_train, y_train, X_test, y_test = trainer.load_data()

    if choice == 1:
        epochs = int(input("Digite o número de épocas para treinamento: "))
        batch_size = int(input("Digite o tamanho do lote (batch size): "))
        validation_split = float(input("Digite a proporção de validação (entre 0 e 1): "))

        trainer.train_model(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        print("Treinamento concluído.")

    elif choice == 2:
        trainer.evaluate_model(X_test, y_test)

        # Realizar previsão para uma nova amostra
        new_sample = [float(input(f"Digite o valor do atributo {i + 1}: ")) for i in range(4)]
        predicted_class = trainer.predict(new_sample)
        print(f"Classe prevista: {predicted_class}")
    
    elif choice == 3:
        print(trainer.check_device())
        
    elif choice == 4:
        matrix_size = int(input("Digite o tamanho da matriz: "))
        trainer.gpu_benchmark(matrix_size=matrix_size)

    else:
        print("Opção inválida. Escolha 1 ou 2.")

if __name__ == "__main__":
    main()

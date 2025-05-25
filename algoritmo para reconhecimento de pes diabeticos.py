# ped_classifier.py
# Projeto acadêmico: Classificação de pés diabéticos com EfficientNetB3

# Importações necessárias
import os  # manipulação de diretórios
import argparse  # leitura de argumentos via terminal
import numpy as np  # operações numéricas
import pandas as pd  # manipulação de dados em tabela
import matplotlib.pyplot as plt  # gráficos de linha
from sklearn.metrics import confusion_matrix, classification_report as sk_classification_report  # métricas de avaliação
from tensorflow import keras  # biblioteca principal de deep learning
from tensorflow.keras import regularizers  # regularização das camadas
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # aumento de dados
from tensorflow.keras.applications import EfficientNetB3  # backbone pré-treinado
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout  # camadas customizadas
from tensorflow.keras.models import Model  # construção de modelo
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint  # callbacks úteis
import seaborn as sns  # gráficos de calor para matriz de confusão

# Classe para carregar e preparar os dados
class DataPreprocessor:
    def __init__(self, base_dir, img_size=(300, 300), batch_size=32):
        self.base_dir = base_dir  # diretório base
        self.img_size = img_size  # tamanho das imagens
        self.batch_size = batch_size  # tamanho do lote

    def create_generators(self):
        # Definição de aumento de dados para o conjunto de treino
        train_aug = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=True,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            validation_split=0.2  # separa parte para validação
        )

        # Geração de dados de treino
        train_gen = train_aug.flow_from_directory(
            directory=self.base_dir,
            target_size=self.img_size,
            class_mode='categorical',
            batch_size=self.batch_size,
            subset='training',
            shuffle=True
        )

        # Geração de dados de validação
        val_gen = train_aug.flow_from_directory(
            directory=self.base_dir,
            target_size=self.img_size,
            class_mode='categorical',
            batch_size=self.batch_size,
            subset='validation',
            shuffle=False
        )
        return train_gen, val_gen


# Classe para construção do modelo baseado no EfficientNetB3
class ModelBuilder:
    def __init__(self, input_shape, n_classes, l2=1e-4, dropout=0.5):
        self.input_shape = input_shape  # formato da imagem
        self.n_classes = n_classes  # número de classes de saída
        self.l2 = l2  # fator de regularização
        self.dropout = dropout  # taxa de dropout

    def build(self):
        # Carrega a EfficientNetB3 sem as camadas finais
        base = EfficientNetB3(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape,
            pooling='avg'  # aplica global average pooling
        )
        base.trainable = False  # congela os pesos base

        # Adiciona normalização por lotes
        x = BatchNormalization()(base.output)

        # Camada densa com ativação e regularização
        x = Dense(
            256, activation='relu',
            kernel_regularizer=regularizers.l2(self.l2)
        )(x)

        # Dropout para evitar overfitting
        x = Dropout(self.dropout)(x)

        # Camada de saída softmax para classificação multiclasse
        outputs = Dense(self.n_classes, activation='softmax')(x)

        return Model(inputs=base.input, outputs=outputs)


# Classe para treinar o modelo
class Trainer:
    def __init__(self, model, train_gen, val_gen, epochs=30, lr=1e-3, out_dir='output'):
        self.model = model  # modelo a ser treinado
        self.train_gen = train_gen  # dados de treino
        self.val_gen = val_gen  # dados de validação
        self.epochs = epochs  # número de épocas
        self.lr = lr  # taxa de aprendizado
        self.out_dir = out_dir  # diretório de saída
        os.makedirs(self.out_dir, exist_ok=True)  # cria diretório se necessário

    def compile(self):
        # Compila o modelo com otimizador Adam e função de perda categórica
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self):
        # Define os callbacks de treinamento
        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
            EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
            ModelCheckpoint(
                filepath=os.path.join(self.out_dir, 'best_model.h5'),
                monitor='val_loss', save_best_only=True, verbose=1)
        ]

        # Inicia o treinamento
        return self.model.fit(
            self.train_gen,
            validation_data=self.val_gen,
            epochs=self.epochs,
            callbacks=callbacks
        )


# Classe para avaliar o modelo
class Evaluator:
    @staticmethod
    def plot_history(history):
        # Extrai métricas do histórico de treino
        acc, val_acc = history.history['accuracy'], history.history['val_accuracy']
        loss, val_loss = history.history['loss'], history.history['val_loss']
        epochs = range(1, len(acc) + 1)

        # Plota a acurácia
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, label='Train Acc')
        plt.plot(epochs, val_acc, label='Val Acc')
        plt.title('Acurácia')
        plt.xlabel('Época')
        plt.ylabel('Acurácia')
        plt.legend()

        # Plota o loss
        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, label='Train Loss')
        plt.plot(epochs, val_loss, label='Val Loss')
        plt.title('Loss')
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def confusion_matrix(y_true, y_pred, class_names):
        # Calcula e plota matriz de confusão com valores
        cm = confusion_matrix(y_true, y_pred, labels=class_names)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
        plt.title('Matriz de Confusão')
        plt.ylabel('Verdadeiro')
        plt.xlabel('Previsto')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def classification_report(y_true, y_pred, class_names):
        # Verifica se há previsões
        if not y_true or not y_pred:
            print("Aviso: vetores de entrada vazios. Nenhuma previsão realizada.")
            return
        # Imprime o relatório com precisão, revocação e F1
        report = sk_classification_report(
            y_true, y_pred, labels=class_names, target_names=class_names, zero_division=0
        )
        print("Relatório de Classificação:\n", report)


# Classe para realizar predições usando modelo salvo
class Predictor:
    def __init__(self, model_path, class_indices_csv, img_size=(300, 300)):
        df = pd.read_csv(class_indices_csv)  # carrega mapeamento de classes
        self.class_names = df['class'].tolist()  # nomes das classes
        self.model = keras.models.load_model(model_path)  # carrega modelo treinado
        self.img_size = img_size  # tamanho da imagem

    def predict_folder(self, folder):
        y_true, y_pred = [], []  # listas para resultados
        for class_name in os.listdir(folder):
            class_folder = os.path.join(folder, class_name)
            if not os.path.isdir(class_folder):
                continue
            for file in os.listdir(class_folder):
                if not file.lower().endswith(('jpg', 'jpeg', 'png')):
                    continue
                fpath = os.path.join(class_folder, file)
                img = keras.preprocessing.image.load_img(fpath, target_size=self.img_size)
                x = keras.preprocessing.image.img_to_array(img) / 255.0
                x = np.expand_dims(x, axis=0)
                probs = self.model.predict(x, verbose=0)[0]  # predição
                pred_label = self.class_names[np.argmax(probs)]  # rótulo previsto
                y_true.append(class_name)  # rótulo verdadeiro
                y_pred.append(pred_label)  # rótulo previsto
                print(f"{file} -> Real: {class_name}, Previsto: {pred_label} ({np.max(probs)*100:.2f}%)")
        return y_true, y_pred


# Função principal de execução
def main(args):
    # Carrega os dados
    train_gen, val_gen = DataPreprocessor(
        base_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size
    ).create_generators()

    # Salva o mapeamento das classes
    index_to_class = {v: k for k, v in train_gen.class_indices.items()}
    pd.DataFrame(index_to_class.items(), columns=["index", "class"]).to_csv(
        os.path.join(args.output, 'class_indices.csv'), index=False
    )

    # Constrói o modelo
    model = ModelBuilder(
        input_shape=(args.img_size, args.img_size, 3),
        n_classes=train_gen.num_classes
    ).build()

    # Treina o modelo
    trainer = Trainer(model, train_gen, val_gen, epochs=args.epochs, out_dir=args.output)
    trainer.compile()
    history = trainer.train()

    # Avalia desempenho durante treino
    Evaluator.plot_history(history)

    # Prediz e avalia nos dados de teste
    predictor = Predictor(
        model_path=os.path.join(args.output, 'best_model.h5'),
        class_indices_csv=os.path.join(args.output, 'class_indices.csv'),
        img_size=(args.img_size, args.img_size)
    )
    y_true, y_pred = predictor.predict_folder(args.test_dir)
    Evaluator.confusion_matrix(y_true, y_pred, predictor.class_names)
    Evaluator.classification_report(y_true, y_pred, predictor.class_names)


# Execução via terminal
if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # define argumentos CLI
    parser.add_argument('--data_dir', type=str, required=True)  # pasta de dados
    parser.add_argument('--test_dir', type=str, required=True)  # pasta de teste
    parser.add_argument('--output', type=str, default='output')  # saída
    parser.add_argument('--img_size', type=int, default=300)  # tamanho da imagem
    parser.add_argument('--batch_size', type=int, default=32)  # tamanho do lote
    parser.add_argument('--epochs', type=int, default=30)  # épocas
    args = parser.parse_args()  # analisa os argumentos
    main(args)  # executa

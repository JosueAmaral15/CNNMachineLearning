# ped_classifier.py
# Projeto acadêmico: Classificação de pés diabéticos com EfficientNetB3

import os  # operações de sistema de arquivos
import argparse  # processamento de argumentos de linha de comando
import numpy as np  # cálculos numéricos
import pandas as pd  # manipulação de dados
import matplotlib.pyplot as plt  # visualização de resultados
from sklearn.metrics import confusion_matrix, classification_report, recall_score, precision_score
from tensorflow import keras  # framework principal
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint


class DataPreprocessor:
    """
    Classe para preparar dados de treino, validação e teste.
    """
    def __init__(self, base_dir, img_size=(300, 300), batch_size=32):
        # diretório raiz dos dados e configurações
        self.base_dir = base_dir
        self.img_size = img_size
        self.batch_size = batch_size

    def create_generators(self):
        # gerador de aumento de dados para treino
        train_aug = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=True,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            validation_split=0.2  # separa 20% para validação
        )
        # gerador sem aumento para validação e teste
        test_aug = ImageDataGenerator(rescale=1./255)

        # fluxo de imagens para treino
        train_gen = train_aug.flow_from_directory(
            directory=self.base_dir,
            target_size=self.img_size,
            class_mode='categorical',
            batch_size=self.batch_size,
            subset='training',  # usa partição de treino
            shuffle=True
        )
        # fluxo de imagens para validação
        val_gen = train_aug.flow_from_directory(
            directory=self.base_dir,
            target_size=self.img_size,
            class_mode='categorical',
            batch_size=self.batch_size,
            subset='validation',  # usa partição de validação
            shuffle=False
        )
        return train_gen, val_gen, test_aug


class ModelBuilder:
    """
    Classe para construir o modelo EfficientNetB3.
    """
    def __init__(self, input_shape, n_classes, l2=1e-4, dropout=0.5):
        # formato da entrada e hiper parâmetros
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.l2 = l2
        self.dropout = dropout

    def build(self):
        # base pré-treinada sem topo
        base = EfficientNetB3(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape,
            pooling='avg'  # GlobalAveragePooling
        )
        # congela pesos iniciais
        base.trainable = False

        # adiciona camadas de classificação
        x = BatchNormalization()(base.output)  # normalização de features
        x = Dense(
            256,
            activation='relu',
            kernel_regularizer=regularizers.l2(self.l2),
            bias_regularizer=regularizers.l2(self.l2)
        )(x)
        x = Dropout(self.dropout)(x)  # evita overfitting
        outputs = Dense(self.n_classes, activation='softmax')(x)

        model = Model(inputs=base.input, outputs=outputs)
        return model


class Trainer:
    """
    Classe para treinar o modelo com callbacks e métricas.
    """
    def __init__(self, model, train_gen, val_gen, epochs=30, lr=1e-3, out_dir='output'):
        self.model = model
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.epochs = epochs
        self.lr = lr
        self.out_dir = out_dir
        # cria diretório de saída
        os.makedirs(self.out_dir, exist_ok=True)

    def compile(self):
        # compila com otimizador Adam e loss multiclass
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self):
        # callback para reduzir LR ao estagnar 'val_loss'
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, verbose=1
        )
        # callback para early stopping
        early_stop = EarlyStopping(
            monitor='val_loss', patience=5, verbose=1, restore_best_weights=True
        )
        # callback para salvar melhor modelo
        checkpoint = ModelCheckpoint(
            filepath=os.path.join(self.out_dir, 'best_model.h5'),
            monitor='val_loss', save_best_only=True, verbose=1
        )

        # executa treinamento
        history = self.model.fit(
            self.train_gen,
            epochs=self.epochs,
            validation_data=self.val_gen,
            callbacks=[reduce_lr, early_stop, checkpoint]
        )
        return history


class Evaluator:
    """
    Classe para avaliação de desempenho: plots e métricas.
    """
    @staticmethod
    def plot_history(history):
        # extrai métricas do histórico
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)

        # plota curvas de acurácia
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, label='Train Acc')
        plt.plot(epochs, val_acc, label='Val Acc')
        plt.title('Acurácia durante o Treino')
        plt.xlabel('Época')
        plt.ylabel('Acurácia')
        plt.legend()

        # plota curvas de loss
        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, label='Train Loss')
        plt.plot(epochs, val_loss, label='Val Loss')
        plt.title('Loss durante o Treino')
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def confusion_matrix(y_true, y_pred, class_names):
        # gera matriz de confusão
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Matriz de Confusão')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    @staticmethod
    def classification_report(y_true, y_pred, class_names):
        # exibe relatório no terminal
        report = classification_report(y_true, y_pred, target_names=class_names)
        print("Classification Report:\n", report)


class Predictor:
    """
    Classe para realizar previsões em novos dados.
    """
    def __init__(self, model_path, class_indices_csv, img_size=(300, 300)):
        # carrega csv de índices de classes
        df = pd.read_csv(class_indices_csv)
        self.class_names = df['class'].tolist()
        # carrega modelo
        self.model = keras.models.load_model(model_path)
        self.img_size = img_size

    def predict_folder(self, folder):
        # coleta todas as imagens
        files = [os.path.join(folder, f) for f in os.listdir(folder)]
        y_true, y_pred = [], []
        for fpath in files:
            # extrai label verdadeiro do nome da pasta
            true_label = os.path.basename(os.path.dirname(fpath))
            img = keras.preprocessing.image.load_img(
                fpath, target_size=self.img_size
            )
            x = keras.preprocessing.image.img_to_array(img) / 255.0
            x = np.expand_dims(x, axis=0)
            probs = self.model.predict(x)[0]
            pred_idx = np.argmax(probs)
            pred_label = self.class_names[pred_idx]
            y_true.append(true_label)
            y_pred.append(pred_label)
            # exibe no terminal
            print(f"{os.path.basename(fpath)} -> True: {true_label}, Pred: {pred_label} ({probs[pred_idx]*100:.2f}%)")
        # retorna vetores para análise
        return y_true, y_pred


def main(args):
    # inicializa pré-processamento
    dp = DataPreprocessor(
        base_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size
    )
    
    train_gen, val_gen, _ = dp.create_generators()

    # Salva o mapeamento de classes para arquivo CSV
    class_indices = train_gen.class_indices
    index_to_class = {v: k for k, v in class_indices.items()}
    df_classes = pd.DataFrame(list(index_to_class.items()), columns=['index', 'class'])
    df_classes.to_csv(os.path.join(args.output, 'class_indices.csv'), index=False)


    # constrói modelo
    mb = ModelBuilder(
        input_shape=(args.img_size, args.img_size, 3),
        n_classes=train_gen.num_classes,
        l2=1e-4,
        dropout=0.5
    )
    model = mb.build()

    # treina
    tr = Trainer(
        model=model,
        train_gen=train_gen,
        val_gen=val_gen,
        epochs=args.epochs,
        lr=1e-3,
        out_dir=args.output
    )
    tr.compile()
    history = tr.train()

    # avalia histórico e gera métricas
    Evaluator.plot_history(history)

    # exemplo de predição e matriz de confusão (utilização acadêmica)
    pred = Predictor(
        model_path=os.path.join(args.output, 'best_model.h5'),
        class_indices_csv=os.path.join(args.output, 'class_indices.csv'),
        img_size=(args.img_size, args.img_size)
    )
    y_true, y_pred = pred.predict_folder(args.test_dir)
    # gera matriz de confusão e relatório
    Evaluator.confusion_matrix(y_true, y_pred, pred.class_names)
    Evaluator.classification_report(y_true, y_pred, pred.class_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Treina e avalia modelo EfficientNetB3 para pés diabéticos'
    )
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Diretório com subpastas de classes para treino e validação')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Diretório com subpastas de classes para testes finais')
    parser.add_argument('--output', type=str, default='output',
                        help='Pasta para salvar modelos e índices de classe')
    parser.add_argument('--img_size', type=int, default=300,
                        help='Tamanho de redimensionamento das imagens (quadrado)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Tamanho do batch para treinamento')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Número máximo de épocas de treinamento')
    args = parser.parse_args()
    main(args)
    
'''

python3 "algoritmo para reconhecimento de pes diabeticos.py" \
  --data_dir "/home/josue/Documentos/Acadêmico/Mestrado UFF/Aprendizado de máquina/trabalho final/data/DFU/Patches" \
  --test_dir "/home/josue/Documentos/Acadêmico/Mestrado UFF/Aprendizado de máquina/trabalho final/test" \
  --output "./output" \
  --img_size 300 \
  --batch_size 32 \
  --epochs 40

'''
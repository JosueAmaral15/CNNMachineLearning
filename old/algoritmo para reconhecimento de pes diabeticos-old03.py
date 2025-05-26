# ped_classifier.py
# Projeto acadêmico: Classificação de pés diabéticos com EfficientNetB3

import os  # operações de sistema de arquivos
import argparse  # processamento de argumentos de linha de comando
import numpy as np  # cálculos numéricos
import pandas as pd  # manipulação de dados
import matplotlib.pyplot as plt  # visualização de resultados
from sklearn.metrics import confusion_matrix, classification_report as sk_classification_report, recall_score, precision_score
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
        self.base_dir = base_dir
        self.img_size = img_size
        self.batch_size = batch_size

    def create_generators(self):
        train_aug = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=True,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            validation_split=0.2
        )
        test_aug = ImageDataGenerator(rescale=1./255)

        train_gen = train_aug.flow_from_directory(
            directory=self.base_dir,
            target_size=self.img_size,
            class_mode='categorical',
            batch_size=self.batch_size,
            subset='training',
            shuffle=True
        )
        val_gen = train_aug.flow_from_directory(
            directory=self.base_dir,
            target_size=self.img_size,
            class_mode='categorical',
            batch_size=self.batch_size,
            subset='validation',
            shuffle=False
        )
        return train_gen, val_gen, test_aug


class ModelBuilder:
    def __init__(self, input_shape, n_classes, l2=1e-4, dropout=0.5):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.l2 = l2
        self.dropout = dropout

    def build(self):
        base = EfficientNetB3(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape,
            pooling='avg'
        )
        base.trainable = False

        x = BatchNormalization()(base.output)
        x = Dense(
            256,
            activation='relu',
            kernel_regularizer=regularizers.l2(self.l2),
            bias_regularizer=regularizers.l2(self.l2)
        )(x)
        x = Dropout(self.dropout)(x)
        outputs = Dense(self.n_classes, activation='softmax')(x)

        model = Model(inputs=base.input, outputs=outputs)
        return model


class Trainer:
    def __init__(self, model, train_gen, val_gen, epochs=30, lr=1e-3, out_dir='output'):
        self.model = model
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.epochs = epochs
        self.lr = lr
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def compile(self):
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self):
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
        checkpoint = ModelCheckpoint(
            filepath=os.path.join(self.out_dir, 'best_model.h5'),
            monitor='val_loss', save_best_only=True, verbose=1
        )

        history = self.model.fit(
            self.train_gen,
            epochs=self.epochs,
            validation_data=self.val_gen,
            callbacks=[reduce_lr, early_stop, checkpoint]
        )
        return history


class Evaluator:
    @staticmethod
    def plot_history(history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, label='Train Acc')
        plt.plot(epochs, val_acc, label='Val Acc')
        plt.title('Acurácia durante o Treino')
        plt.xlabel('Época')
        plt.ylabel('Acurácia')
        plt.legend()

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
        cm = confusion_matrix(y_true, y_pred, labels=class_names)
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
        labels = sorted(list(set(y_true) | set(y_pred)))
        report = sk_classification_report(y_true, y_pred, labels=labels, target_names=class_names, zero_division=0)
        print("Classification Report:\n", report)


class Predictor:
    def __init__(self, model_path, class_indices_csv, img_size=(300, 300)):
        df = pd.read_csv(class_indices_csv)
        self.class_names = df['class'].tolist()
        self.model = keras.models.load_model(model_path)
        self.img_size = img_size

    def predict_folder(self, folder):
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('jpg', 'png', 'jpeg'))]
        y_true, y_pred = [], []
        for fpath in files:
            true_label = os.path.basename(os.path.dirname(fpath))
            img = keras.preprocessing.image.load_img(fpath, target_size=self.img_size)
            x = keras.preprocessing.image.img_to_array(img) / 255.0
            x = np.expand_dims(x, axis=0)
            probs = self.model.predict(x)[0]
            pred_idx = np.argmax(probs)
            pred_label = self.class_names[pred_idx]
            y_true.append(true_label)
            y_pred.append(pred_label)
            print(f"{os.path.basename(fpath)} -> True: {true_label}, Pred: {pred_label} ({probs[pred_idx]*100:.2f}%)")
        return y_true, y_pred


def main(args):
    dp = DataPreprocessor(
        base_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size
    )
    train_gen, val_gen, _ = dp.create_generators()

    # salva mapeamento de classes
    class_indices = train_gen.class_indices
    index_to_class = {v: k for k, v in class_indices.items()}
    df_classes = pd.DataFrame(list(index_to_class.items()), columns=['index', 'class'])
    df_classes.to_csv(os.path.join(args.output, 'class_indices.csv'), index=False)

    mb = ModelBuilder(
        input_shape=(args.img_size, args.img_size, 3),
        n_classes=train_gen.num_classes,
        l2=1e-4,
        dropout=0.5
    )
    model = mb.build()

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

    Evaluator.plot_history(history)

    pred = Predictor(
        model_path=os.path.join(args.output, 'best_model.h5'),
        class_indices_csv=os.path.join(args.output, 'class_indices.csv'),
        img_size=(args.img_size, args.img_size)
    )
    y_true, y_pred = pred.predict_folder(args.test_dir)
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
  --test_dir "/home/josue/Documentos/Acadêmico/Mestrado UFF/Aprendizado de máquina/trabalho final/data/DFU/Original Images" \
  --output "./output" \
  --img_size 300 \
  --batch_size 32 \
  --epochs 100

'''
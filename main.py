import os
import numpy as np
import tensorflow as tf
from keras.layers import Activation, Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from transformers import BertConfig, TFBertModel, BertTokenizer

from label import Label
from sample import Sample

max_length = 16
configuration = BertConfig()
save_path = "Model/"
data_path = "Data/"

if not os.path.exists(save_path):
    os.makedirs(save_path)

if not os.path.exists(save_path + "vocab.txt"):
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    tokenizer.save_pretrained(save_path)

tokenizer = BertTokenizer(save_path + "vocab.txt", strip_accents=False)


# Preprocess given dataset
def preprocess(file, tokenizer):
    samples = []
    categories = set()
    labels = set()

    with open(file, "r", encoding="utf-8") as input_file:
        for line in input_file:

            # Lines with comments
            if line[0] == '#':
                continue

            line = line.rstrip()

            # Empty line
            if not line:
                continue

            # Parse sample
            parts = line.split("\t")
            word = parts[1]
            base = parts[2]
            category = parts[3]
            token = tokenizer.tokenize(word)
            label = parts[4]
            specification = parts[5]

            # Save samples, categories and labels
            samples.append(Sample(word, base, category, token, label, specification))
            categories.add(category)
            labels.add(label)

    return samples, categories, labels


# Load saved model or download new one
def load_model():
    model = TFBertModel.from_pretrained("bert-base-multilingual-cased")
    model.save_pretrained(save_path)

    return model


# Train model with given parameters
# Bert used as the embedding
# TODO compare sparse_categorical_crossentropy vs categorical_crossentropy
def train_model(x, y, num_labels, epochs=5, batch_size=128):
    bert = load_model()
    inputs = Input(shape=(max_length,), dtype=tf.int32)
    embedding = bert(inputs)[1]
    outputs = Dense(num_labels, activation='softmax')(embedding)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.summary()
    model.fit(x=x, y=y, epochs=epochs, batch_size=batch_size, verbose=1)

    return model


# Predict values from model
def predict(model: Model, x_test, y_test):

    index = model.predict(x_test)
    print(y_test[0])
    print(index)

    return model.predict(x_test)


# Encode label to one hot vector
def encode_label(label, num_labels):
    encoded = [0] * num_labels
    encoded[label] = 1
    return encoded


# Prepares training data
def prepare_training_data():
    samples, categories, labels = preprocess(data_path + "dev.txt", tokenizer)
    label2id = {i: label for i, label in enumerate(labels)}
    id2label = dict(zip(label2id.values(), label2id.keys()))
    num_labels = len(labels)

    x = []
    y = []

    for sample in samples:
        encoded_token = tokenizer.encode(sample.token)
        encoded_label = encode_label(id2label[sample.label.text], num_labels)
        padding_length = max_length - len(encoded_token)

        # Add padding to align to max length
        if padding_length > 0:
            encoded_token = encoded_token + ([0] * padding_length)

        x.append(encoded_token)
        y.append(encoded_label)

    return np.array(x), np.array(y), labels, num_labels


def main():
    print("Loading data")
    x_train, y_train, labels, num_labels = prepare_training_data()
    print("Training")
    model = train_model(x_train, y_train, num_labels)
    test = predict(model, x_train, y_train)


if __name__ == "__main__":
    main()

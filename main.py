import logging
import os
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Dropout, Input
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from transformers import BertConfig, BertTokenizer, TFBertModel, TFBertForTokenClassification

from sentence import Sentence
from utils import prepare_data_for_end2end
from word import Word


# Key constants
label_parts = 15
max_length = 64 # 128 better
special_tokens_count = 2
pad_token_id = 0
pad_token_label = "X"
pad_decomposed_token = [0] * label_parts

# Preparation of BERT configuration and tokenization
configuration = BertConfig("bert-base-multilingual-cased")
model_name = "tf_model.h5"
save_path = "Model/"

if not os.path.exists(save_path):
    os.makedirs(save_path)

if not os.path.exists(save_path + "vocab.txt"):
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    tokenizer.save_pretrained(save_path)

tokenizer = BertTokenizer(save_path + "vocab.txt", do_lower_case=False, strip_accents=True)


# Preprocess given dataset
def preprocess(file_name):
    labels = []
    sentences = []
    tokens = []
    words = []
    categories = set()
    unique_labels = set()
    text = str()

    with open(file_name, "r", encoding="utf-8") as input_file:
        for line in input_file:

            # Lines with comments
            if line[0] == '#':
                continue

            line = line.rstrip()

            # Empty line between sentences
            if not line:
                sentences.append(Sentence(text, tokens.copy(), labels.copy()))
                text = str()
                tokens.clear()
                labels.clear()
                continue

            # Parse one word
            parts = line.split("\t")
            word = parts[1]
            base = parts[2]
            category = parts[3]
            token = tokenizer.tokenize(word)
            label = parts[4]
            specification = parts[5]

            # Skip invalid label
            if len(label) < label_parts:
                continue

            # Update sentence text, tokens and labels
            text = text + " " + word
            tokens.extend(token)
            labels.extend([label] * (len(token)))

            # Save samples, categories and labels
            words.append(Word(word, base, category, token, label, specification))
            categories.add(category)
            unique_labels.add(label)

    return sentences, categories, unique_labels, words


# Load saved model or download new one
def load_model(num_labels, trainable=True):
    # Use BertForTokenClassification
    model = TFBertForTokenClassification.from_pretrained("bert-base-multilingual-cased", num_labels=num_labels)
    model.trainable = trainable

    return model


# Train model with given parameters and save result
# Bert used as the embedding
def train_model(x, y, am, num_labels, epochs=3, batch_size=64):
    model = load_model(num_labels)
    inputs = Input(shape=(max_length,), dtype=tf.int32, name="input_words_ids")
    masks = Input(shape=(max_length,), dtype=tf.int32, name="input_mask_ids")

    model([inputs, masks])
    model.layers[-1].activation = tf.keras.activations.softmax
    model.compile(optimizer=Adam(lr=3e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(x=[x, am], y=y, epochs=epochs, batch_size=batch_size, verbose=1)
    model.save_pretrained(save_path + "composed/")

    return model


# Train model with given parameters for prediction of decomposed label
# TFBertModel used as the embedding layer of multiple dense output layers
def train_decomposed_model(x, y, am, num_labels, epochs=2, batch_size=8):
    K.clear_session() # Clear GPU memory 
    transformer = TFBertModel.from_pretrained("bert-base-multilingual-cased")
    inputs = Input(shape=(max_length,), dtype=tf.int32, name='input_words_ids')
    masks = Input(shape=(max_length,), dtype=tf.int32, name='input_mask_ids')

    # Build all 15 outputs one for each part of label
    outputs = []
    for i in range(label_parts):
        layer = transformer([inputs, masks])[0]
        layer = Dropout(0.2)(layer)
        layer = Dense(units=num_labels[i])(layer)
        outputs.append(layer)
    
    model = Model(inputs=[inputs, masks], outputs=outputs)
    model.compile(optimizer=Adam(lr=2e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(x=[x, am], y=y, epochs=epochs, batch_size=batch_size, verbose=1)
    #model.save(save_path + "decomposed/")

    return model


# Trains end-to-end model which inputs are from two different models
# Input size of the model is 16 due to 15 parts of the decomposed label plus one composed prediction
def train_end2end_model(x, y, num_labels, epochs=250, batch_size=1024):
    K.clear_session() # Clear GPU memory 
    inputs = Input(shape=(max_length, 16, ), name='label_id')
    layer = Dense(units=32, activation='relu')(inputs)
    layer = Dropout(0.1)(layer)
    layer = Dense(units=64, activation='relu')(layer)
    layer = Dropout(0.1)(layer)
    layer = Dense(units=128, activation='relu')(layer)
    layer = Dropout(0.1)(layer)
    layer = Dense(units=256, activation='relu')(layer)
    layer = Dropout(0.1)(layer)
    layer = Dense(units=512, activation='relu')(layer)
    layer = Dropout(0.1)(layer)
    outputs = Dense(units=num_labels, activation='softmax')(layer)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(x=x, y=y, epochs=epochs, batch_size=batch_size, verbose=1)
    model.save(save_path + "end2end/")

    return model


# Evaluates given model
# Computes, accuracy, precision, recall and f-score
def evaluate_model(model, x_test, am_test, y_test, id2label):
    result = model.evaluate([x_test, am_test], y_test)
    print(dict(zip(model.metrics_names, result)))

    y_pred = model.predict([x_test, am_test], batch_size=64, verbose=1)
    y_pred = np.argmax(y_pred.logits, axis=2)

    unique_classes = np.sort(np.unique(np.concatenate((y_test, y_pred))))
    target_names = [id2label[i] for i in unique_classes]
    print(classification_report(y_test.flatten(), y_pred.flatten(), target_names=target_names))


# Prepare decomposed labels for each sentence
# Returns id2label, label2id and number of labels for each part
def prepare_decomposed_labels(sentences):
    decomposed_labels = [set() for _ in range(label_parts)]
    id2label, label2id, num_labels = [], [], []

    for sentence in sentences:
        for label in sentence.get_labels_decomposed():
            for i, part in enumerate(label):
                decomposed_labels[i].add(part)

    # Create id2label and label2id dictionaries
    for i, labels in enumerate(decomposed_labels):
        id2label.append({i: label for i, label in enumerate(labels)})
        label2id.append(dict(zip(id2label[i].values(), id2label[i].keys())))
        num_labels.append(len(labels))

    return label2id, num_labels


# Encodes all parts of decomposed label
def encode_decomposed_label(sentence, label2id):
    encoded_labels = []

    # Encode individual parts
    for label in sentence.get_labels_decomposed():
        encoded_parts = []
        for i, part in enumerate(label):
            encoded_parts.append(label2id[i].get(part))
        encoded_labels.append(encoded_parts)

    # Solve case when the length of decomposed labels is greater than max length
    if len(encoded_labels) > max_length - special_tokens_count:
        encoded_labels = encoded_labels[:(max_length - special_tokens_count)]

    # Add padding to the decomposed labels
    encoded_labels = [pad_decomposed_token] + encoded_labels + [pad_decomposed_token]
    if len(encoded_labels) < max_length:
        encoded_labels = encoded_labels + [[0] * label_parts] * (max_length - len(encoded_labels))

    return encoded_labels


# Prepares training data
def prepare_data(filename):
    sentences, categories, labels, words = preprocess(filename)
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = dict(zip(id2label.values(), id2label.keys()))
    num_labels = len(labels)
    d_label2id, d_num_labels = prepare_decomposed_labels(sentences)

    x = []
    attention_masks = []
    y = []
    y_decomposed = []

    for sentence in sentences:
        encoded_label = []
        # Padding is done using encode_plus method
        encoded_token = tokenizer.encode_plus(sentence.tokens,
                                              add_special_tokens=True,
                                              return_attention_mask=True,
                                              return_token_type_ids=True,
                                              truncation=True,
                                              padding="max_length",
                                              max_length=max_length)
        # Encode sentence labels
        for label in sentence.labels:
            encoded_label.append(label2id[label])

        # Encode decomposed labels of the sentence
        encoded_decomposed_label = encode_decomposed_label(sentence, d_label2id)

        if len(encoded_label) > max_length - special_tokens_count:
            encoded_label = encoded_label[:(max_length - special_tokens_count)]

        # Add padding to labels
        encoded_label = [pad_token_id] + encoded_label + [pad_token_id]
        if len(encoded_label) < max_length:
            encoded_label = encoded_label + [0] * (max_length - len(encoded_label))

        # Append encoded token, attention mask and label
        x.append(encoded_token['input_ids'])
        attention_masks.append(encoded_token['attention_mask'])
        y.append(encoded_label.copy())
        y_decomposed.append(encoded_decomposed_label.copy())

    return np.array(x), np.array(y), np.array(y_decomposed), np.array(attention_masks), \
           labels, num_labels, d_num_labels, id2label


def main():
    logging.basicConfig(format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO)
    logging.info("Loading data.")

    x, y, y_decomposed, attention_masks, labels, num_labels, d_num_labels, id2label = prepare_data("Data/train.txt")
    x_train, x_test, y_train, y_test, yd_train, yd_test, am_train, am_test = \
        train_test_split(x, y, y_decomposed, attention_masks, test_size=0.2)

    # Fix dimensions of decomposed labels
    yd_train = np.split(yd_train, label_parts, axis=2)
    yd_test = np.split(yd_test, label_parts, axis=2)

    logging.info("Loading finished. Data prepared.")
    logging.info(f"Loaded {len(x)} examples, total {num_labels} labels.")
    logging.info(f"Training model for composed label with {len(x_train)} examples, total {num_labels} labels.")
    model = train_model(x_train, y_train, am_train, num_labels)

    logging.info("Training finished. Evaluate model.")
    result = model.evaluate([x_test, am_test], y_test)
    print(dict(zip(model.metrics_names, result)))
    
    logging.info("Predicting values for end2end model.")
    y_pred = model.predict([x_train, am_train], batch_size=128, verbose=1)
    y_pred = np.argmax(y_pred.logits, axis=2)

    logging.info(f"Training model for decomposed label.")
    model = train_decomposed_model(x_train, yd_train, am_train, d_num_labels)

    logging.info("Training finished. Evaluate model.")
    result = model.evaluate([x_test, am_test], yd_test)
    print(dict(zip(model.metrics_names, result)))
    
    logging.info("Preparing data for end2end model.")
    yd_pred = model.predict([x_train, am_train], batch_size=128, verbose=1)
    y_pred = prepare_data_for_end2end(y_pred, yd_pred)

    logging.info(f"Training end2end model.")
    model = train_end2end_model(y_pred, y_train, num_labels)

    logging.info("Training finished. Evaluate model.")
    x_test = np.concatenate((yd_test, y_test[..., None]))
    result = model.evaluate(x_test, y_test)
    print(dict(zip(model.metrics_names, result)))


if __name__ == "__main__":
    main()

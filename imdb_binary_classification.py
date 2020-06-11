'''A two-class classification or binary classiciation trained on the IMDB dataset from keras.
It will attempt to classify a movie review as positive or negative.'''

from keras.datasets import imdb
from keras import models, layers
import numpy as np
import matplotlib.pyplot as plt

# decode the list of integers, which are indices of words back to words
def decode_to_english(data):
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    # indices are offset by 3 because 0, 1, and 2 are reserved indices for 'padding', 'start of sequence' and 'unknown'
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in data])
    return decoded_review


# vectorizing the list of integers into a tensor: a 10,000 dimensional vector
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


def plot_loss(history):
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc = history_dict['accuracy']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


def main():
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    # vectorize the labels
    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')
    
    # create a validation set to monitor the training
    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]
    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]


    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
    print(history.history.keys())
    plot_loss(history)

    # print(decode_to_english(train_data[1]))
    # print(train_labels[1])

if __name__ == '__main__':
    main()
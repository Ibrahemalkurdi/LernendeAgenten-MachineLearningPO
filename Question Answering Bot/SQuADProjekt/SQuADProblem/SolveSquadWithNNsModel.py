import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt

import json
import sys

print(sys.getdefaultencoding())
with open('train-v2.0.json', 'r') as f:
    json_file = json.load(f)
question_context_list = []
# answer_start_list_of_lists = []
# answer_end_list_of_lists = []
answer_start_end_list_of_lists = []
for title_paragraphs in json_file['data']:
    for qas_context in title_paragraphs['paragraphs']:
        for qas in qas_context['qas']:
            question_context = qas['question'] + " : " + qas_context['context']
            question_context_list.append(qas['question'] + ":" + qas_context['context'])
            if len(qas['answers']) == 0:
                # answer_start_list_of_lists.append([])
                # answer_end_list_of_lists.append([])
                answer_start_end_list_of_lists.append([-1, -1])
            else:
                # answers_start = []
                # answers_end = []
                answers_start_end = []
                for answer in qas['answers']:
                    # answers_start.append(answer['answer_start'])
                    answers_start_end.append(answer['answer_start'])
                    answers_start_end.append(answer['answer_start'] + len(answer['text']) - 1)
                    # if answer['answer_start'] > max:
                    # max = answer['answer_start']
                    # answers_end.append(answer['answer_start'] + len(answer['text']) - 1)
                # answer_start_list_of_lists.append(answers_start)
                # answer_end_list_of_lists.append(answers_end)
                answer_start_end_list_of_lists.append(answers_start_end)
print(max)
training_samples = 100000
tokenizer = Tokenizer()
tokenizer.fit_on_texts(question_context_list)
sequences = tokenizer.texts_to_sequences(question_context_list)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences)
# data = np.asarray(sequences)
# answers_start = np.asarray(answer_start_list_of_lists)
# answers_end = np.asarray(answer_end_list_of_lists)
answers_start_end = np.asarray(answer_start_end_list_of_lists)
print('Shape of data tensor:', data.shape)  # Shape of data tensor: (130319,)
print(data.shape[1])
print('Shape of answers tensor:', answers_start_end.shape)  # Shape of answers tensor: (130319,)
# print(data_np)
# print(answers)

# Split the data into a training set and a validation set
# But first, shuffle the data
indices = np.arange(data.shape[0])
# print(indices.shape)
np.random.shuffle(indices)
data = data[indices]
answers_start_end = answers_start_end[indices]
# answers_start = answers_start[indices]
# answers_end = answers_end[indices]

x_train = data[:training_samples]
y_train = answers_start_end[:training_samples]
x_val = data[training_samples:]
y_val = answers_start_end[training_samples:]
print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)

glove_dir = '../glove.6B'

embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

embedding_dim = 100
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    # print(word)
    # print(i)
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
model = Sequential()
model.add(Embedding(len(word_index) + 1, embedding_dim, input_length=data.shape[1]))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='relu'))
model.summary()

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

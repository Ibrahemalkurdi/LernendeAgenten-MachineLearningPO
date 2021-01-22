import os
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, activations, models, preprocessing
from tensorflow.keras import preprocessing, utils
import re
import matplotlib.pyplot as plt

print(tf.__version__)

dir_path = '..\\QADataset'
files_list = os.listdir(dir_path + os.sep)

batch_size = 12  # Batch size for training.
epochs = 1  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.

questions = list()
answers = list()

for filepath in files_list:
    stream = open(dir_path + os.sep + filepath, 'rb')
    docs = yaml.safe_load(stream)
    conversations = docs['conversations']
    for con in conversations:
        if len(con) > 2:
            questions.append(con[0])
            replies = con[1:]
            ans = ''
            for rep in replies:
                ans += ' ' + rep
            answers.append(ans)
        elif len(con) > 1:
            questions.append(con[0])
            answers.append(con[1])

answers_with_tags = list()
for i in range(len(answers)):
    if type(answers[i]) == str:
        answers_with_tags.append(answers[i])
    else:
        questions.pop(i)

answers = list()
for i in range(len(answers_with_tags)):
    answers.append('<START> ' + answers_with_tags[i] + ' <END>')

print(len(answers))
print(len(questions))


def tokenize(sentences):
    # tokens_list = []
    # vocabulary = []
    sentences_clear = []
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = re.sub(' \'', ' ', sentence)
        sentence = re.sub('can\'t', 'can not', sentence)
        sentence = re.sub('n\'t', ' not', sentence)
        sentence = re.sub('\'ve', ' have', sentence)
        sentence = re.sub('\'ll', ' will', sentence)
        sentence = re.sub('\'s', ' is', sentence)
        sentence = re.sub('\'m', ' am', sentence)
        sentence = re.sub('\'re', ' are', sentence)
        sentence = re.sub('\'d', ' would', sentence)
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)
        sentences_clear.append(sentence)
        # tokens = sentence.split()
        # vocabulary += tokens
        # tokens_list.append(tokens)
    return sentences_clear


answers = tokenize(answers)
questions = tokenize(questions)

tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(questions + answers)
# sequences_answers = tokenizer.texts_to_sequences(answers)
word_index = tokenizer.word_index
VOCAB_SIZE = len(word_index) + 1
print('Found %s unique tokens.' % VOCAB_SIZE)  # Found 1841 unique tokens.

embedding_dim = 100

# shuffle
indices = np.arange(len(questions))
np.random.shuffle(indices)

# encoder_input_data
tokenized_questions = tokenizer.texts_to_sequences(questions)
maxlen_questions = max([len(x) for x in tokenized_questions])
encoder_input_data = preprocessing.sequence.pad_sequences(tokenized_questions, maxlen=maxlen_questions, padding='post')
encoder_input_data = encoder_input_data[indices]
print(encoder_input_data.shape, maxlen_questions)

# decoder_input_data
tokenized_answers = tokenizer.texts_to_sequences(answers)
maxlen_answers = max([len(x) for x in tokenized_answers])
decoder_input_data = preprocessing.sequence.pad_sequences(tokenized_answers, maxlen=maxlen_answers, padding='post')
decoder_input_data = decoder_input_data[indices]
print(decoder_input_data.shape, maxlen_answers)

# decoder_output_data
tokenized_answers = tokenizer.texts_to_sequences(answers)
for i in range(len(tokenized_answers)):
    tokenized_answers[i] = tokenized_answers[i][1:]
padded_answers = preprocessing.sequence.pad_sequences(tokenized_answers, maxlen=maxlen_answers, padding='post')
decoder_output_data = padded_answers[indices]
decoder_output_data = utils.to_categorical(decoder_output_data, VOCAB_SIZE)
print(decoder_output_data.shape)

encoder_inputs = tf.keras.layers.Input(shape=(None,))
encoder_embedding = tf.keras.layers.Embedding(VOCAB_SIZE, embedding_dim, mask_zero=True)(encoder_inputs)
encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(latent_dim, dropout=0.2, recurrent_dropout=0.2,
                                                         return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = tf.keras.layers.Input(shape=(None,))
decoder_embedding = tf.keras.layers.Embedding(VOCAB_SIZE, embedding_dim, mask_zero=True)(decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM(latent_dim, dropout=0.2, recurrent_dropout=0.2, return_state=True,
                                    return_sequences=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = tf.keras.layers.Dense(VOCAB_SIZE, activation=tf.keras.activations.softmax)
output = decoder_dense(decoder_outputs)

model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output)
model.summary()
tf.keras.utils.plot_model(model, to_file='modelWithoutPreTrEmb.png', show_shapes=True)

model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='categorical_crossentropy', metrics=['acc'])

history = model.fit([encoder_input_data, decoder_input_data], decoder_output_data, batch_size=batch_size, epochs=epochs,
                    validation_split=0.1)
model.save('modelWithoutPreTrEmb.h5')

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
# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)

decoder_state_input_h = tf.keras.layers.Input(shape=(latent_dim,))
decoder_state_input_c = tf.keras.layers.Input(shape=(latent_dim,))

decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = tf.keras.models.Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_word_index = dict(
    (i, word) for word, i in word_index.items())


def decode_sequence(input_seq):
    tokens = str_to_tokens(input_seq)
    if tokens is None:
        return None
    # Encode the input as state vectors.
    states_value = encoder_model.predict(tokens)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))
    # Populate the first word of target sequence with the start word.
    target_seq[0][0] = word_index['start']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_word_index[sampled_token_index]
        decoded_sentence += ' ' + sampled_word

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_word == 'end' or
                len(decoded_sentence.split()) > maxlen_answers):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0][0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence


def str_to_tokens(sentence: str):
    words = sentence.split()
    tokens_list = list()
    for word in words:
        if word in word_index:
            tokens_list.append(word_index[word])
    if len(tokens_list) == 0:
        return None
    else:
        return preprocessing.sequence.pad_sequences([tokens_list], maxlen=maxlen_questions, padding='post')


exit_pro = False
while not exit_pro:
    input_seq = input('Enter question or exit : ')
    input_seq = input_seq.lower().strip()
    if input_seq == "exit":
        exit_pro = True
    else:
        decoded_sentence = decode_sequence(input_seq)
        if decoded_sentence is None:
            print("Sorry, I can't answer this question")
        else:
            print(decoded_sentence)

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

batch_size = 40  # Batch size for training.
epochs = 1  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.

input_texts = []
target_texts = []
for filepath in files_list:
    stream = open(dir_path + os.sep + filepath, 'rb')
    docs = yaml.safe_load(stream)
    conversations = docs['conversations']
    for con in conversations:
        if len(con) > 2:
            input_texts.append(con[0])
            replies = con[1:]
            ans = ''
            for rep in replies:
                ans += ' ' + rep
            target_texts.append(ans)
        elif len(con) > 1:
            input_texts.append(con[0])
            target_texts.append(con[1])

answers_with_tags = list()
for i in range(len(target_texts)):
    if type(target_texts[i]) == str:
        answers_with_tags.append(target_texts[i])
    else:
        input_texts.pop(i)

target_texts = list()
for i in range(len(answers_with_tags)):
    target_texts.append('\t' + answers_with_tags[i] + '\n')


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
        sentence = re.sub('[^a-zA-Z\t\n\\.,]', ' ', sentence)
        sentences_clear.append(sentence)
        # tokens = sentence.split()
        # vocabulary += tokens
        # tokens_list.append(tokens)
    return sentences_clear


target_texts = tokenize(target_texts)
input_texts = tokenize(input_texts)
input_characters = set()
target_characters = set()
for i in range(len(input_texts)):
    for char in input_texts[i]:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_texts[i]:
        if char not in target_characters:
            target_characters.add(char)
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))

num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)

max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.
    decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.
    decoder_target_data[i, t:, target_token_index[' ']] = 1.

# shuffle
indices = np.arange(len(input_texts))
np.random.shuffle(indices)

encoder_input_data = encoder_input_data[indices]
decoder_input_data = decoder_input_data[indices]
decoder_target_data = decoder_target_data[indices]
# Define an input sequence and process it.
encoder_inputs = tf.keras.layers.Input(shape=(None, num_encoder_tokens))
encoder = tf.keras.layers.LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = tf.keras.layers.Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = tf.keras.layers.LSTM(latent_dim, return_sequences=True
                                    , return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)

decoder_dense = tf.keras.layers.Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = tf.keras.models.Model([encoder_inputs, decoder_inputs]
                              , decoder_outputs)
tf.keras.utils.plot_model(model, to_file='modelCharByChar.png', show_shapes=True)
model.summary()
# Run training
model.compile(optimizer=tf.keras.optimizers.RMSprop()
              , loss='categorical_crossentropy', metrics=['acc'])

history = model.fit([encoder_input_data, decoder_input_data]
                    , decoder_target_data
                    , batch_size=batch_size
                    , epochs=epochs,
                    validation_split=0.1)

# Save model
model.save('modelCharByChar.h5')

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
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = tf.keras.models.Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
                len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


def str_to_tokens(sentence: str):
    input_seq = np.zeros((1, max_encoder_seq_length, num_encoder_tokens),
                         dtype='float32')
    for i in range(min(len(sentence), max_encoder_seq_length)):
        input_seq[0, i, input_token_index[sentence[i]]] = 1.
    input_seq[0, i + 1:, input_token_index[' ']] = 1.
    if len(input_seq) == 0:
        return None
    else:
        return input_seq


exit_pro = False
while not exit_pro:
    input_seq = input('Enter question or exit : ')
    input_seq = input_seq.lower().strip()
    if input_seq == "exit":
        exit_pro = True
    else:
        input_seq = str_to_tokens(input_seq)
        if input_seq is None:
            print("Sorry, I can't answer this question")
        else:
            decoded_sentence = decode_sequence(input_seq)
            print(decoded_sentence)
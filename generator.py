import tensorflow as tf
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.text import hashing_trick
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
import numpy as np
nltk.download('stopwords')
from tensorflow.keras.preprocessing.text import hashing_trick


def load_model(file):
    return tf.keras.models.load_model(file, compile=False)


class Madochan():

    def __init__(self, model = None):

        if not model:
            model = load_model('./madochan/models/1000epochs600lenhashingbidirectional.h5')
        self.change_model(model)
        self.voc_size = 5000 #Vocabulary size for hashing trick
        self.ps=PorterStemmer()
        self.max_decoder_seq_length = 37 #Maximum word size
        self.weirdness = 1 #Determines which prediction to use. 1 = argmax, 2 equals second best prediction, etc. The higher, the more the words become weird.
        self.reverse_target_char_index = dict({0: '\t', 1: '\n', 2: ' ', 3: '-', 4: 'a', 5: 'b', 6: 'c', 7: 'd', 8: 'e', 9: 'f', 10: 'g', 11: 'h', 12: 'i', 13: 'j', 14: 'k', 15: 'l', 16: 'm', 17: 'n', 18: 'o', 19: 'p', 20: 'q', 21: 'r', 22: 's', 23: 't', 24: 'u', 25: 'v', 26: 'w', 27: 'x', 28: 'y', 29: 'z'})
        self.reverse_input_char_index = dict({0: '\t', 1: '\n', 2: ' ', 3: '!', 4: '"', 5: '#', 6: '$', 7: '%', 8: '&', 9: "'", 10: '(', 11: ')', 12: '*', 13: '+', 14: ',', 15: '-', 16: '.', 17: '/', 18: '0', 19: '1', 20: '2', 21: '3', 22: '4', 23: '5', 24: '6', 25: '7', 26: '8', 27: '9', 28: ':', 29: ';', 30: '=', 31: '>', 32: '@', 33: 'A', 34: 'B', 35: 'C', 36: 'D', 37: 'E', 38: 'F', 39: 'G', 40: 'H', 41: 'I', 42: 'J', 43: 'K', 44: 'L', 45: 'M', 46: 'N', 47: 'O', 48: 'P', 49: 'Q', 50: 'R', 51: 'S', 52: 'T', 53: 'U', 54: 'V', 55: 'W', 56: 'X', 57: 'Y', 58: 'Z', 59: '[', 60: '\\', 61: ']', 62: '^', 63: '_', 64: '`', 65: 'a', 66: 'b', 67: 'c', 68: 'd', 69: 'e', 70: 'f', 71: 'g', 72: 'h', 73: 'i', 74: 'j', 75: 'k', 76: 'l', 77: 'm', 78: 'n', 79: 'o', 80: 'p', 81: 'q', 82: 'r', 83: 's', 84: 't', 85: 'u', 86: 'v', 87: 'w', 88: 'x', 89: 'y', 90: 'z', 91: '{', 92: '|', 93: '}', 94: '~', 95: '£', 96: '§', 97: '°', 98: 'º', 99: '¼', 100: '½', 101: '¾', 102: '¿', 103: 'Æ', 104: 'É', 105: 'Ü', 106: 'Þ', 107: 'à', 108: 'á', 109: 'â', 110: 'ä', 111: 'å', 112: 'æ', 113: 'ç', 114: 'è', 115: 'é', 116: 'ê', 117: 'ë', 118: 'ì', 119: 'î', 120: 'ï', 121: 'ð', 122: 'ñ', 123: 'ò', 124: 'ó', 125: 'ô', 126: 'ö', 127: '÷', 128: 'ù', 129: 'û', 130: 'ü', 131: 'þ'})
        self.encoder_model = None
        self.decoder_model = None

    def change_model(self,newmodel):

        self.model = newmodel
        self.longest_sentence_len = newmodel.input_shape[0][1]
        self.latent_dim = newmodel.layers[-2].output[0].shape[-1]
        self.num_decoder_tokens = newmodel.input_shape[1][2]

    def create_encoder_decoder(self):

        self.encoder_inputs = self.model.input[0]  # input_1

        if len(self.model.layers[3].output) == 5:
            self.encoder_outputs, self.state_h_enc, self.state_c_enc, _, _ = self.model.layers[3].output  # One directional

        elif len(self.model.layers[3].output) == 3:
            self.encoder_outputs, self.state_h_enc, self.state_c_enc = self.model.layers[3].output  # Bidirectional

        self.encoder_states = [self.state_h_enc, self.state_c_enc]
        self.encoder_model = tf.keras.Model(self.encoder_inputs, self.encoder_states)

        self.decoder_inputs = self.model.input[1]
        self.decoder_state_input_h = tf.keras.Input(shape=(self.latent_dim,), name = 'state_h')
        self.decoder_state_input_c = tf.keras.Input(shape=(self.latent_dim,), name = 'state_c')
        self.decoder_states_inputs = [self.decoder_state_input_h, self.decoder_state_input_c]
        self.decoder_lstm = self.model.layers[4]
        self.decoder_outputs, self.state_h_dec, self.state_c_dec = self.decoder_lstm(self.decoder_inputs, initial_state=self.decoder_states_inputs)
        self.decoder_states = [self.state_h_dec, self.state_c_dec]
        self.decoder_dense = self.model.layers[5]
        self.decoder_outputs = self.decoder_dense(self.decoder_outputs)
        self.decoder_model = tf.keras.Model([self.decoder_inputs] + self.decoder_states_inputs, [self.decoder_outputs] + self.decoder_states)

    def preprocess(self,message):

        review=re.sub("[^a-zA-Z]"," ", str(message))
        review = re.sub(r'http\S+', " ", str(review))
        review = re.sub(r'@\w+',' ', str(review))
        review = re.sub(r'#\w+', ' ', str(review))
        review = re.sub('r<.*?>',' ', str(review))
        review=review.lower()
        review=review.split()
        review=[self.ps.stem(word) for word in review if not word in stopwords.words("english")]
        review=" ".join(review)

        hashed_string = hashing_trick(review,self.voc_size, 'md5') #Using md5 for consistency between runs

        embedded_docs=pad_sequences([hashed_string],padding="post",maxlen=self.longest_sentence_len).squeeze()

        return embedded_docs

    def decode_sequence(self, input_seq):

        self.create_encoder_decoder()

        #We wrap the predictions with a tf.function to avoid retracing warnings.
        self.encoder_model.predict_function = tf.function(experimental_relax_shapes=True)(self.encoder_model)
        self.decoder_model.predict_function = tf.function(experimental_relax_shapes=True)(self.decoder_model)

        states_value = self.encoder_model.predict_function(input_seq)
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))

        stop_condition = False
        decoded_sentence = ""
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict_function([target_seq] + states_value)

            sampled_token_index = np.argsort(output_tokens[0, -1, :])[-self.weirdness]
            sampled_char = self.reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            if sampled_char == "\n" or len(decoded_sentence) > self.max_decoder_seq_length:
                stop_condition = True

            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.0

            # Update states
            states_value = [h, c]
        return decoded_sentence

    def create_word(self, definition):
        preprocessed = self.preprocess(definition)
        input_seq = np.expand_dims(preprocessed,0)
        decoded_sentence = self.decode_sequence(input_seq)

        return decoded_sentence

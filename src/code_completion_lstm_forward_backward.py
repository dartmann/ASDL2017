from builtins import print

import numpy
import tensorflow as tf
import tflearn


class Code_Completion_Forward_Backward:
    def token_to_string(self, token):
        return token["type"] + "-@@-" + token["value"]

    def string_to_token(self, string):
        splitted = string.split("-@@-")
        return {"type": splitted[0], "value": splitted[1]}

    def one_hot(self, string):
        vector = [0] * len(self.string_to_number)
        vector[self.string_to_number[string]] = 1
        return vector

    def prepare_data(self, token_lists):
        # encode tokens into one-hot vectors
        all_token_strings = set()
        for token_list in token_lists:
            for token in token_list:
                all_token_strings.add(self.token_to_string(token))
        # Fixes the low accuracy bug when loading model from file
        all_token_strings = list(all_token_strings)
        all_token_strings.sort()
        print("Unique tokens: " + str(len(all_token_strings)))
        self.string_to_number = dict()
        self.number_to_string = dict()
        max_number = 0
        for token_string in all_token_strings:
            self.string_to_number[token_string] = max_number
            self.number_to_string[max_number] = token_string
            max_number += 1
        # Prepare forward backward x,y pairs
        xf = yf = xb = yb = []
        for token_list in token_lists:
            for idx, token in enumerate(token_list):
                self.create_seq(idx, token, token_list, xf, yf)
            # Reverse the token list
            token_list = token_list[::-1]
            for idx, token in enumerate(token_list):
                self.create_seq(idx, token, token_list, xb, yb)
        print("x,y pairs: " + str(len(xf)) + ", " + str(len(yf)) + ", " + str(len(xb)) + ", " + str(len(yb)))
        return (xf, yf, xb, yb)

    # Helper which is called in the two for loops of forward and backward enumeration in #prepare_data
    def create_seq(self, idx, token, token_list, xs, ys):
        if idx > 3:
            prev1 = self.token_to_string(token_list[idx - 1])
            prev2 = self.token_to_string(token_list[idx - 2])
            prev3 = self.token_to_string(token_list[idx - 3])
            prev4 = self.token_to_string(token_list[idx - 4])
            seq_forward_or_backward = [self.one_hot(prev1), self.one_hot(prev2), self.one_hot(prev3),
                                       self.one_hot(prev4)]
            xs.append(seq_forward_or_backward)
        else:
            xs.append(self.one_hot(self.token_to_string(token_list[idx - 1])))
        ys.append(self.one_hot(self.token_to_string(token)))

    def create_network(self):
        # Setup forward network
        self.net_forward = tflearn.input_data(shape=[None, 4, len(self.string_to_number)])
        self.net_forward = tflearn.lstm(self.net_forward, 128, return_seq=True)
        self.net_forward = tflearn.lstm(self.net_forward, 32, return_seq=True)
        self.net_forward = tflearn.lstm(self.net_forward, 32, return_seq=True)
        self.net_forward = tflearn.lstm(self.net_forward, 128)
        self.net_forward = tflearn.fully_connected(self.net_forward, len(self.string_to_number), activation='softmax')
        self.net_forward = tflearn.regression(self.net_forward)
        self.model_forward = tflearn.DNN(self.net_forward)
        # Else we would get an error
        tf.reset_default_graph()
        # Setup backward network
        self.net_backward = tflearn.input_data(shape=[None, 4, len(self.string_to_number)])
        self.net_backward = tflearn.lstm(self.net_backward, 128, return_seq=True)
        self.net_backward = tflearn.lstm(self.net_backward, 32, return_seq=True)
        self.net_backward = tflearn.lstm(self.net_backward, 32, return_seq=True)
        self.net_backward = tflearn.lstm(self.net_backward, 128)
        self.net_backward = tflearn.fully_connected(self.net_backward, len(self.string_to_number), activation='softmax')
        self.net_backward = tflearn.regression(self.net_backward)
        self.model_backward = tflearn.DNN(self.net_backward)

    def load(self, token_lists, model_file):
        self.prepare_data(token_lists)
        self.create_network()
        self.model_forward.load("./../trained_model/forward_model")
        tf.reset_default_graph()
        self.model_backward.load("./../trained_model/backward_model")

    def train(self, token_lists, model_file):
        (xf, yf, xb, yb) = self.prepare_data(token_lists)
        self.create_network()
        reshaped_xf = numpy.reshape(xf, (-1, 4, len(self.string_to_number)))
        reshaped_xb = numpy.reshape(xb, (-1, 4, len(self.string_to_number)))
        self.model_forward.fit(reshaped_xf, yf, n_epoch=1, batch_size=1024, show_metric=True)
        self.model_backward.fit(reshaped_xb, yb, n_epoch=1, batch_size=1024, show_metric=True)
        self.model_forward.save("./../trained_model/forward_model")
        self.model_backward.save("./../trained_model/backward_model")

    def query(self, prefix, suffix):
        forward_seq = self.query_forward_or_backward(prefix, True)
        backward_seq = self.query_forward_or_backward(suffix, False)
        predicted_seq = [a + b for a, b in zip(forward_seq, backward_seq)]
        best_number = predicted_seq.index(max(predicted_seq))
        best_string = self.number_to_string[best_number]
        best_token = self.string_to_token(best_string)
        return [best_token]

    # Helper for forward and backward prediction
    def query_forward_or_backward(self, prefix_or_suffix, is_forward):
        if not is_forward:
            prefix_or_suffix = prefix_or_suffix[::-1]
        prev4 = self.one_hot(self.token_to_string(prefix_or_suffix[-1]))
        prev3 = prev2 = prev1 = [0] * len(self.string_to_number)
        if len(prefix_or_suffix) > 3:
            prev3 = self.one_hot(self.token_to_string(prefix_or_suffix[-2]))
            prev2 = self.one_hot(self.token_to_string(prefix_or_suffix[-3]))
            prev1 = self.one_hot(self.token_to_string(prefix_or_suffix[-4]))
        elif len(prefix_or_suffix) > 2:
            prev3 = self.one_hot(self.token_to_string(prefix_or_suffix[-2]))
            prev2 = self.one_hot(self.token_to_string(prefix_or_suffix[-3]))
        elif len(prefix_or_suffix) > 1:
            prev3 = self.one_hot(self.token_to_string(prefix_or_suffix[-2]))
        seq_forward_or_backward = [prev1, prev2, prev3, prev4]
        x = numpy.reshape(seq_forward_or_backward, (-1, 4, len(self.string_to_number)))
        if is_forward:
            y = self.model_forward.predict(x)
        else:
            y = self.model_backward.predict(x)
        predicted_seq = y[0]
        if type(predicted_seq) is numpy.ndarray:
            predicted_seq = predicted_seq.tolist()
        return predicted_seq

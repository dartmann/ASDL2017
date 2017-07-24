import tflearn
import numpy
import tensorflow as tf

# one hole = 0.61
# two holes = 0.35
# three holes = 0.
class Code_Completion_Test:
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

        # prepare x,y pairs
        xf = []
        yf = []

        xb = []
        yb = []
        for token_list in token_lists:
            for idx, token in enumerate(token_list):
                sequence = []
                if idx > 3:
                    token_string = self.token_to_string(token)
                    previous4 = self.token_to_string(token_list[idx - 1])
                    previous3 = self.token_to_string(token_list[idx - 2])
                    previous2 = self.token_to_string(token_list[idx - 3])
                    previous1 = self.token_to_string(token_list[idx - 4])

                    sequence.append(self.one_hot(previous1))
                    sequence.append(self.one_hot(previous2))
                    sequence.append(self.one_hot(previous3))
                    sequence.append(self.one_hot(previous4))
                    xf.append(sequence)
                    yf.append(self.one_hot(token_string))

        for token_list in token_lists:
            token_list = token_list[::-1]
            for idx, token in enumerate(token_list):
                sequence = []
                if idx > 3:
                    token_string = self.token_to_string(token)
                    previous4 = self.token_to_string(token_list[idx - 1])
                    previous3 = self.token_to_string(token_list[idx - 2])
                    previous2 = self.token_to_string(token_list[idx - 3])
                    previous1 = self.token_to_string(token_list[idx - 4])

                    sequence.append(self.one_hot(previous1))
                    sequence.append(self.one_hot(previous2))
                    sequence.append(self.one_hot(previous3))
                    sequence.append(self.one_hot(previous4))
                    xb.append(sequence)
                    yb.append(self.one_hot(token_string))

        print("x,y pairs: " + str(len(xf)) + " " + str(len(yf)) + " " + str(len(xb)) + " " + str(len(yb)))
        return (xf, yf, xb, yb)

    def create_network(self):
        self.netForward = tflearn.input_data(shape=[None, 4, len(self.string_to_number)])
        self.netForward = tflearn.lstm(self.netForward, 128, return_seq=True)
        self.netForward = tflearn.lstm(self.netForward, 32, return_seq=True)
        self.netForward = tflearn.lstm(self.netForward, 32, return_seq=True)
        self.netForward = tflearn.lstm(self.netForward, 128)
        self.netForward = tflearn.fully_connected(self.netForward, len(self.string_to_number), activation='softmax')
        self.netForward = tflearn.regression(self.netForward)
        self.modelForward = tflearn.DNN(self.netForward)

        tf.reset_default_graph()

        self.netBackwards = tflearn.input_data(shape=[None, 4, len(self.string_to_number)])
        self.netBackwards = tflearn.lstm(self.netBackwards, 128, return_seq=True)
        self.netBackwards = tflearn.lstm(self.netBackwards, 32, return_seq=True)
        self.netBackwards = tflearn.lstm(self.netBackwards, 32, return_seq=True)
        self.netBackwards = tflearn.lstm(self.netBackwards, 128)
        self.netBackwards = tflearn.fully_connected(self.netBackwards, len(self.string_to_number), activation='softmax')
        self.netBackwards = tflearn.regression(self.netBackwards)
        self.modelBackwards = tflearn.DNN(self.netBackwards)

    def load(self, token_lists, model_file):
        self.prepare_data(token_lists)
        self.create_network()
        self.modelForward.load("./../model/forwardModel")
        tf.reset_default_graph()
        self.modelBackwards.load("./../model/backwardModel")

    def train(self, token_lists, model_file):
        (xf, yf, xb, yb) = self.prepare_data(token_lists)
        print("x,y pairs: " + str(len(xf)) + " " + str(len(yf)) + " " + str(len(xb)) + " " + str(len(yb)))
        self.create_network()

        XF = numpy.reshape(xf, (-1, 4, len(self.string_to_number)))
        XB = numpy.reshape(xb, (-1, 4, len(self.string_to_number)))
        self.modelBackwards.fit(XB, yb, n_epoch=1, batch_size=1024, show_metric=True)
        self.modelForward.fit(XF, yf, n_epoch=1, batch_size=1024, show_metric=True)

        self.modelForward.save("./../model/forwardModel")
        self.modelBackwards.save("./../model/backwardModel")

    def query(self, prefix, suffix):

        predicted_forward_sequence = self.queryForward(prefix)
        predicted_backwards_sequence = self.queryBackwards(suffix)
        predicted_seq = [a + b for a, b in zip(predicted_forward_sequence, predicted_backwards_sequence)]
        best_number = predicted_seq.index(max(predicted_seq))
        best_string = self.number_to_string[best_number]
        best_token = self.string_to_token(best_string)
        return [best_token]

    def queryBackwards(self, suffix):
        suffix = suffix[::-1]
        if len(suffix) > 0:
            previous4 = self.one_hot(self.token_to_string(suffix[-1]))
            sequence = []
            if len(suffix) > 3:
                previous3 = self.one_hot(self.token_to_string(suffix[-2]))
                previous2 = self.one_hot(self.token_to_string(suffix[-3]))
                previous1 = self.one_hot(self.token_to_string(suffix[-4]))
            elif len(suffix) > 2:
                previous3 = self.one_hot(self.token_to_string(suffix[-2]))
                previous2 = self.one_hot(self.token_to_string(suffix[-3]))
                previous1 = [0] * len(self.number_to_string)
            elif len(suffix) > 1:
                previous3 = self.one_hot(self.token_to_string(suffix[-2]))
                previous2 = [0] * len(self.string_to_number)
                previous1 = [0] * len(self.string_to_number)
            else:
                previous3 = [0] * len(self.string_to_number)
                previous2 = [0] * len(self.string_to_number)
                previous1 = [0] * len(self.string_to_number)

            sequence.append(previous1)
            sequence.append(previous2)
            sequence.append(previous3)
            sequence.append(previous4)
            x = numpy.reshape(sequence, (-1, 4, len(self.string_to_number)))
            y = self.modelBackwards.predict(x)
            predicted_seq = y[0]
            if type(predicted_seq) is numpy.ndarray:
                predicted_seq = predicted_seq.tolist()
        else:
            predicted_seq = [0] * len(self.string_to_number)
        return predicted_seq

    def queryForward(self, prefix):
        previous4 = self.one_hot(self.token_to_string(prefix[-1]))
        sequence = []
        if len(prefix) > 3:
            previous3 = self.one_hot(self.token_to_string(prefix[-2]))
            previous2 = self.one_hot(self.token_to_string(prefix[-3]))
            previous1 = self.one_hot(self.token_to_string(prefix[-4]))
        elif len(prefix) > 2:
            previous3 = self.one_hot(self.token_to_string(prefix[-2]))
            previous2 = self.one_hot(self.token_to_string(prefix[-3]))
            previous1 = [0] * len(self.number_to_string)
        elif len(prefix) > 1:
            previous3 = self.one_hot(self.token_to_string(prefix[-2]))
            previous2 = [0] * len(self.string_to_number)
            previous1 = [0] * len(self.string_to_number)
        else:
            previous3 = [0] * len(self.string_to_number)
            previous2 = [0] * len(self.string_to_number)
            previous1 = [0] * len(self.string_to_number)

        sequence.append(previous1)
        sequence.append(previous2)
        sequence.append(previous3)
        sequence.append(previous4)
        x = numpy.reshape(sequence, (-1, 4, len(self.string_to_number)))
        y = self.modelForward.predict(x)
        predicted_seq = y[0]
        if type(predicted_seq) is numpy.ndarray:
            predicted_seq = predicted_seq.tolist()
        return predicted_seq



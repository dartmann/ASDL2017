import numpy
import tflearn


class Code_Completion_Lstm:

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
        # TODO: here we can implement the 2nd dim with 3
        #   -> build a sequence (arr) which gets the 3 previous tokens and then is fed into xs. Thats all...
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

        # prepare x,y pairs
        xs = []
        ys = []
        for token_list in token_lists:
            for idx, token in enumerate(token_list):
                if idx > 0:
                    token_string = self.token_to_string(token)
                    previous_token_string = self.token_to_string(token_list[idx - 1])
                    # Create the initial one hot encoding vector (before if-else, because both depend on it)
                    v = [0] * len(self.string_to_number)
                    # If we are not the last token
                    if (idx + 1) < len(token_list):
                        # Get the next token
                        s = self.token_to_string(token_list[idx + 1])
                        v = [0] * len(self.string_to_number)
                        # Suffix gets value 3 in vector AND previous token string gets value 2
                        v[self.string_to_number[s]] = 3
                        v[self.string_to_number[previous_token_string]] = 2
                    else:
                        v[self.string_to_number[previous_token_string]] = 2
                    # We are not the first AND the second token
                    if idx > 1:
                        sec_previous_token_string = self.token_to_string(token_list[idx - 2])
                        v[self.string_to_number[sec_previous_token_string]] = 1
                    # If we are not the second last AND the last token
                    if (idx + 2) < len(token_list):
                        s = self.token_to_string(token_list[idx + 2])
                        v[self.string_to_number[s]] = 4
                    #xs.append(self.one_hot(previous_token_string))
                    xs.append(v)
                    ys.append(self.one_hot(token_string))
        print("x,y pairs: " + str(len(xs)))
        return (xs, ys)

    def create_network(self):
        # We reshape the input data to fit 3D, by inserting another dimension (1)
        # TODO: here we can implement the 2nd dim with 3
        self.net = tflearn.input_data(shape=[None, 1, len(self.string_to_number)])
        self.net = tflearn.lstm(self.net, 128, return_seq=True)
        self.net = tflearn.lstm(self.net, 32, return_seq=True)
        self.net = tflearn.lstm(self.net, 32, return_seq=True)
        self.net = tflearn.lstm(self.net, 128)
        self.net = tflearn.fully_connected(self.net, len(self.string_to_number), activation='softmax')
        self.net = tflearn.regression(self.net)
        self.model = tflearn.DNN(self.net, tensorboard_verbose=0)

    def load(self, token_lists, model_file):
        self.prepare_data(token_lists)
        self.create_network()
        self.model.load(model_file)

    def train(self, token_lists, model_file):
        (xs, ys) = self.prepare_data(token_lists)
        self.create_network()
        # We need to reshape the model to 3D
        # TODO: here we can implement the 2nd dim with 3
        xs = numpy.reshape(xs, (-1, 1, 86))
        self.model.fit(xs, ys, n_epoch=1, show_metric=True, batch_size=1024)
        self.model.save(model_file)

    def query(self, prefix, suffix):
        #print("\nSuffix:")
        #print(suffix)
        ## This is important to setup at first!
        previous_token_string = self.token_to_string(prefix[-1])
        # Create the initial one hot encoding vector (before if-else, because both depend on it)
        x = [0] * len(self.string_to_number)
        # If we have a suffix
        ## TODO: non existing previous vectors are simply encoded as zero-vectors
        if (len(suffix) > 0):
            right = self.token_to_string(suffix[0])
            x[self.string_to_number[right]] = 3
            x[self.string_to_number[previous_token_string]] = 2
        else:
            x[self.string_to_number[previous_token_string]] = 2
        if len(prefix) >= 2:
            sec_previous_token_number = self.token_to_string(prefix[-2])
            x[self.string_to_number[sec_previous_token_number]] = 1
        if (len(suffix) > 1):
            sec_right = self.token_to_string(suffix[1])
            x[self.string_to_number[sec_right]] = 4
        #print("previous_token_string: " + previous_token_string)

        #x = self.one_hot(previous_token_string)

        #print("self.one_hot(previous_token_string):")
        #print(x)
        # We need to reshape the model to fit 3D
        y = self.model.predict(numpy.reshape([x], (-1, 1, 86)))
        predicted_seq = y[0]
        """
        print("predicted_seqs:")
        ns = ""
        for n in predicted_seq:
            ns += str(round(n, 4))
            ns += ", "
        print(ns)
        """
        if type(predicted_seq) is numpy.ndarray:
            predicted_seq = predicted_seq.tolist()
        best_number = predicted_seq.index(max(predicted_seq))
        best_string = self.number_to_string[best_number]
        best_token = self.string_to_token(best_string)
        return [best_token]
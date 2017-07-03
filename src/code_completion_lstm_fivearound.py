import tflearn
import numpy


class Code_Completion_FiveAround:
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

        # prepare x,y pairs
        xs = []
        ys = []
        for token_list in token_lists:
            for idx, token in enumerate(token_list):
                # If we have 5 tokens before and afterwards we encode the five previous
                # indexes with [-5, -4, -3, -2, -1] and the five following with [1, 2, 3, 4, 5].
                if idx > 4 and (idx + 5) < len(token_list):
                    # Base vector to work on
                    v = [0] * len(self.string_to_number)
                    # Get prefix token, five indexes before
                    prev5 = self.token_to_string(token_list[idx - 5])
                    # Get prefix token, four indexes before
                    prev4 = self.token_to_string(token_list[idx - 4])
                    # Get prefix token, three indexes before
                    prev3 = self.token_to_string(token_list[idx - 3])
                    # Get prefix token, two indexes before
                    prev2 = self.token_to_string(token_list[idx - 2])
                    # Get prefix token, one index before
                    prev1 = self.token_to_string(token_list[idx - 1])

                    # Get suffix token, one index after
                    suff1 = self.token_to_string(token_list[idx + 1])
                    # Get suffix token, two indexes after
                    suff2 = self.token_to_string(token_list[idx + 2])
                    # Get suffix token, three indexes after
                    suff3 = self.token_to_string(token_list[idx + 3])
                    # Get suffix token, four indexes after
                    suff4 = self.token_to_string(token_list[idx + 4])
                    # Get suffix token, four indexes after
                    suff5 = self.token_to_string(token_list[idx + 5])

                    # fill vector accordingly
                    v[self.string_to_number[prev5]] = -5
                    v[self.string_to_number[prev4]] = -4
                    v[self.string_to_number[prev3]] = -3
                    v[self.string_to_number[prev2]] = -2
                    v[self.string_to_number[prev1]] = -1
                    v[self.string_to_number[suff1]] = 1
                    v[self.string_to_number[suff2]] = 2
                    v[self.string_to_number[suff3]] = 3
                    v[self.string_to_number[suff4]] = 4
                    v[self.string_to_number[suff5]] = 5

                    xs.append(v)
                else:
                    prev1 = self.token_to_string(token_list[idx - 1])
                    xs.append(self.one_hot(prev1))
                ys.append(self.one_hot(self.token_to_string(token)))

        print("x,y pairs: " + str(len(xs)))
        return (xs, ys)

    def create_network(self):
        self.net = tflearn.input_data(shape=[None, 1, len(self.string_to_number)])
        self.net = tflearn.lstm(self.net, 128, return_seq=True)
        self.net = tflearn.lstm(self.net, 32, return_seq=True)
        self.net = tflearn.lstm(self.net, 32, return_seq=True)
        self.net = tflearn.lstm(self.net, 128)
        self.net = tflearn.fully_connected(self.net, len(self.string_to_number), activation='softmax')
        self.net = tflearn.regression(self.net)
        self.model = tflearn.DNN(self.net)

    def load(self, token_lists, model_file):
        self.prepare_data(token_lists)
        self.create_network()
        self.model.load(model_file)

    def train(self, token_lists, model_file):
        (xs, ys) = self.prepare_data(token_lists)
        self.create_network()
        xs = numpy.reshape(xs, (-1, 1, 86))
        self.model.fit(xs, ys, n_epoch=1, batch_size=1024, show_metric=True)
        self.model.save(model_file)

    def query(self, prefix, suffix):
        # Built up the vector like prepared
        x = [0] * len(self.string_to_number)
        # Same as above in #prepare_data
        if len(suffix) > 4 and len(prefix) > 4:
            x[self.string_to_number[self.token_to_string(prefix[-5])]] = -5
            x[self.string_to_number[self.token_to_string(prefix[-4])]] = -4
            x[self.string_to_number[self.token_to_string(prefix[-3])]] = -3
            x[self.string_to_number[self.token_to_string(prefix[-2])]] = -2
            x[self.string_to_number[self.token_to_string(prefix[-1])]] = -1
            x[self.string_to_number[self.token_to_string(suffix[0])]] = 1
            x[self.string_to_number[self.token_to_string(suffix[1])]] = 2
            x[self.string_to_number[self.token_to_string(suffix[2])]] = 3
            x[self.string_to_number[self.token_to_string(suffix[3])]] = 4
            x[self.string_to_number[self.token_to_string(suffix[4])]] = 5
        else:
            previous_token_string = self.token_to_string(prefix[-1])
            x = self.one_hot(previous_token_string)

        y = self.model.predict(numpy.reshape([x], (-1, 1, 86)))
        predicted_seq = y[0]
        if type(predicted_seq) is numpy.ndarray:
            predicted_seq = predicted_seq.tolist()
        best_number = predicted_seq.index(max(predicted_seq))
        best_string = self.number_to_string[best_number]
        best_token = self.string_to_token(best_string)
        return [best_token]

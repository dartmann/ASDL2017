import tflearn
import numpy


class Code_Completion_ThreeAround:
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
                # Base vector to work on
                v = [0] * len(self.string_to_number)
                # If we have three tokens before and afterwards we encode the three previous
                # indexes with [-3, -2, -1] and the three following with [1, 2, 3].
                if idx > 2 and (idx + 3) < len(token_list):
                    # Get prefix token, three indexes before
                    v[self.string_to_number[self.token_to_string(token_list[idx - 3])]] = 1  # -3
                    # Get prefix token, two indexes before
                    v[self.string_to_number[self.token_to_string(token_list[idx - 2])]] = 2  # -2
                    # Get prefix token, one index before
                    v[self.string_to_number[self.token_to_string(token_list[idx - 1])]] = 3  # -1
                    # Get suffix token, one index after
                    v[self.string_to_number[self.token_to_string(token_list[idx + 1])]] = 4  # 1
                    # Get suffix token, two indexes after
                    v[self.string_to_number[self.token_to_string(token_list[idx + 2])]] = 5  # 2
                    # Get suffix token, three indexes after
                    v[self.string_to_number[self.token_to_string(token_list[idx + 3])]] = 6  # 3
                    # Set the prepared vector
                    xs.append(v)
                # If we are at one of the first three tokens
                elif idx <= 2 and (idx + 3) < len(token_list):
                    # Suffix is set anyway
                    v[self.string_to_number[self.token_to_string(token_list[idx + 1])]] = 4  # 1
                    v[self.string_to_number[self.token_to_string(token_list[idx + 2])]] = 5  # 2
                    v[self.string_to_number[self.token_to_string(token_list[idx + 3])]] = 6  # 3
                    # First index is set to zero implicitly and 2nd index to -1
                    if idx >= 1:
                        v[self.string_to_number[self.token_to_string(token_list[idx - 1])]] = 3  # -1
                    if idx >= 2:
                        v[self.string_to_number[self.token_to_string(token_list[idx - 2])]] = 2  # -2
                        # v[self.string_to_number[self.token_to_string(token_list[idx - 1])]] = 3  # -1
                    if idx >= 3:
                        v[self.string_to_number[self.token_to_string(token_list[idx - 3])]] = 1  # -3
                        # v[self.string_to_number[self.token_to_string(token_list[idx - 2])]] = 2  # -2
                        # v[self.string_to_number[self.token_to_string(token_list[idx - 1])]] = 3  # -1
                    # Set the prepared vector
                    xs.append(v)
                # If we are at one of the last three tokens
                elif idx > 2 and (idx + 3) >= len(token_list):
                    # Prefix is set anyway
                    v[self.string_to_number[self.token_to_string(token_list[idx - 3])]] = 1  # -3
                    v[self.string_to_number[self.token_to_string(token_list[idx - 2])]] = 2  # -2
                    v[self.string_to_number[self.token_to_string(token_list[idx - 1])]] = 3  # -1
                    # Last index is set to zero implicitly and 2nd last index to 1
                    if (idx + 2) <= len(token_list):
                        v[self.string_to_number[self.token_to_string(token_list[idx + 1])]] = 4  # 1
                    if (idx + 3) <= len(token_list):
                        # v[self.string_to_number[self.token_to_string(token_list[idx + 1])]] = 4  # 1
                        v[self.string_to_number[self.token_to_string(token_list[idx + 2])]] = 5  # 2
                    if (idx + 4) <= len(token_list):
                        # v[self.string_to_number[self.token_to_string(token_list[idx + 1])]] = 4  # 1
                        # v[self.string_to_number[self.token_to_string(token_list[idx + 2])]] = 5  # 2
                        v[self.string_to_number[self.token_to_string(token_list[idx + 3])]] = 6  # 3
                    # Set the prepared vector
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
        if len(prefix) > 2 and len(suffix) > 2:
            x[self.string_to_number[self.token_to_string(prefix[-2])]] = 1  # -3
            x[self.string_to_number[self.token_to_string(prefix[-1])]] = 2  # -2
            x[self.string_to_number[self.token_to_string(prefix[0])]] = 3  # -1
            x[self.string_to_number[self.token_to_string(suffix[0])]] = 4  # 1
            x[self.string_to_number[self.token_to_string(suffix[1])]] = 5  # 2
            x[self.string_to_number[self.token_to_string(suffix[2])]] = 6  # 3
        elif len(prefix) <= 3 and len(suffix) > 2:
            x[self.string_to_number[self.token_to_string(suffix[0])]] = 4  # 1
            x[self.string_to_number[self.token_to_string(suffix[1])]] = 5  # 2
            x[self.string_to_number[self.token_to_string(suffix[2])]] = 6  # 3
            if len(prefix) >= 1:
                x[self.string_to_number[self.token_to_string(prefix[0])]] = 3  # -1
            if len(prefix) >= 2:
                x[self.string_to_number[self.token_to_string(prefix[-1])]] = 2  # -2
                # x[self.string_to_number[self.token_to_string(prefix[0])]] = 3  # -1
            if len(prefix) >= 3:
                x[self.string_to_number[self.token_to_string(prefix[-2])]] = 1  # -3
                # x[self.string_to_number[self.token_to_string(prefix[-1])]] = 2  # -2
                # x[self.string_to_number[self.token_to_string(prefix[0])]] = 3  # -1
        elif len(prefix) > 2 and len(suffix) <= 3:
            x[self.string_to_number[self.token_to_string(prefix[-2])]] = 1  # -3
            x[self.string_to_number[self.token_to_string(prefix[-1])]] = 2  # -2
            x[self.string_to_number[self.token_to_string(prefix[0])]] = 3  # -1
            if len(suffix) >= 1:
                x[self.string_to_number[self.token_to_string(suffix[0])]] = 4  # 1
            if len(suffix) >= 2:
                # x[self.string_to_number[self.token_to_string(suffix[0])]] = 4  # 1
                x[self.string_to_number[self.token_to_string(suffix[1])]] = 5  # 2
            if len(suffix) >= 3:
                # x[self.string_to_number[self.token_to_string(suffix[0])]] = 4  # 1
                # x[self.string_to_number[self.token_to_string(suffix[1])]] = 5  # 2
                x[self.string_to_number[self.token_to_string(suffix[2])]] = 6  # 3
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

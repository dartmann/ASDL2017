import tflearn
import numpy

# For two prefixes and two suffixes, each of length self.string_to_number
fact_input_dim = 4


class C_C_Lstm_Final_TwoAround:
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
        # We have to add the end of sequence token
        all_token_strings.add("EOS")
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
        (x1, y1) = self.prepare_hole_size_one(token_lists)
        print("33%")
        (x2, y2) = self.prepare_hole_size_two(token_lists)
        print("66%")
        (x3, y3) = self.prepare_hole_size_three(token_lists)
        print("100%")
        return numpy.concatenate([x1, x2, x3]), numpy.concatenate([y1, y2, y3])

    # Prepares x-y pairs for hole of size one
    def prepare_hole_size_one(self, token_lists):
        xs = []
        ys = []
        for token_list in token_lists:
            for idx, token in enumerate(token_list):
                # We only process if the index is lower equal to the 2nd last item,
                # else we have nothing to do here, because we have no hole size of 1.
                if idx < len(token_list) - 1:
                    prefix2 = prefix1 = suffix1 = suffix2 = [0] * len(self.string_to_number)
                    # Hole-Token
                    token_str = self.token_to_string(token)
                    # We are not considering the first 2 and the last 2 elements
                    if 1 < idx < len(token_list) - 2:
                        prefix2 = self.one_hot(self.token_to_string(token_list[idx - 2]))
                        prefix1 = self.one_hot(self.token_to_string(token_list[idx - 1]))
                        suffix1 = self.one_hot(self.token_to_string(token_list[idx + 1]))
                        suffix2 = self.one_hot(self.token_to_string(token_list[idx + 2]))
                    elif 1 == idx < len(token_list) - 2:
                        prefix1 = self.one_hot(self.token_to_string(token_list[idx - 1]))
                        suffix1 = self.one_hot(self.token_to_string(token_list[idx + 1]))
                        suffix2 = self.one_hot(self.token_to_string(token_list[idx + 2]))
                    elif 0 == idx < len(token_list) - 2:
                        suffix1 = self.one_hot(self.token_to_string(token_list[idx + 1]))
                        suffix2 = self.one_hot(self.token_to_string(token_list[idx + 2]))
                    # The index is greater 1 and we are at the second last element
                    elif 1 < idx == len(token_list) - 2:
                        prefix2 = self.one_hot(self.token_to_string(token_list[idx - 2]))
                        prefix1 = self.one_hot(self.token_to_string(token_list[idx - 1]))
                        suffix1 = self.one_hot(self.token_to_string(token_list[idx + 1]))
                    # The index is greater 1
                    elif 1 < idx:
                        prefix2 = self.one_hot(self.token_to_string(token_list[idx - 2]))
                        prefix1 = self.one_hot(self.token_to_string(token_list[idx - 1]))
                    x = numpy.concatenate([prefix2, prefix1, suffix1, suffix2])
                    y = numpy.concatenate([self.one_hot(token_str), self.one_hot("EOS"), self.one_hot("EOS")])
                    xs.append(x)
                    ys.append(y)
        return xs, ys

    # Prepares x-y pairs for hole of size two
    def prepare_hole_size_two(self, token_lists):
        xs = []
        ys = []
        for token_list in token_lists:
            for idx, token in enumerate(token_list):
                # We only process if the index is lower equal to the 3rd last item,
                # else we have nothing to do here, because we have no hole size of 2.
                if idx < len(token_list) - 2:
                    prefix2 = prefix1 = suffix1 = suffix2 = [0] * len(self.string_to_number)
                    # Hole-Tokens
                    token_str = self.token_to_string(token_list[idx])
                    next_token_str = self.token_to_string(token_list[idx + 1])
                    # We are not considering the first 2 and the last 3 elements
                    if 1 < idx < len(token_list) - 3:
                        prefix2 = self.one_hot(self.token_to_string(token_list[idx - 2]))
                        prefix1 = self.one_hot(self.token_to_string(token_list[idx - 1]))
                        suffix1 = self.one_hot(self.token_to_string(token_list[idx + 2]))
                        suffix2 = self.one_hot(self.token_to_string(token_list[idx + 3]))
                    elif 1 == idx < len(token_list) - 3:
                        prefix1 = self.one_hot(self.token_to_string(token_list[idx - 1]))
                        suffix1 = self.one_hot(self.token_to_string(token_list[idx + 2]))
                        suffix2 = self.one_hot(self.token_to_string(token_list[idx + 3]))
                    elif 0 == idx < len(token_list) - 3:
                        suffix1 = self.one_hot(self.token_to_string(token_list[idx + 2]))
                        suffix2 = self.one_hot(self.token_to_string(token_list[idx + 3]))
                    elif 1 < idx == len(token_list) - 3:
                        prefix2 = self.one_hot(self.token_to_string(token_list[idx - 2]))
                        prefix1 = self.one_hot(self.token_to_string(token_list[idx - 1]))
                        suffix1 = self.one_hot(self.token_to_string(token_list[idx + 2]))
                    elif 1 < idx:
                        prefix2 = self.one_hot(self.token_to_string(token_list[idx - 2]))
                        prefix1 = self.one_hot(self.token_to_string(token_list[idx - 1]))
                    x = numpy.concatenate([prefix2, prefix1, suffix1, suffix2])
                    y = numpy.concatenate([self.one_hot(token_str), self.one_hot(next_token_str), self.one_hot("EOS")])
                    xs.append(x)
                    ys.append(y)
        return (xs, ys)

    # Prepares x-y pairs for hole of size three
    def prepare_hole_size_three(self, token_lists):
        xs = []
        ys = []
        for token_list in token_lists:
            for idx, token in enumerate(token_list):
                # We only process if the index is lower equal to the 4th last item,
                # else we have nothing to do here, because we have no hole size of 3.
                if idx < len(token_list) - 3:
                    prefix2 = prefix1 = suffix1 = suffix2 = [0] * len(self.string_to_number)
                    # Hole-Tokens
                    token_str = self.token_to_string(token_list[idx])
                    next_token_str = self.token_to_string(token_list[idx + 1])
                    second_next_token_str = self.token_to_string(token_list[idx + 2])
                    # We are not considering the first 2 and the last 4 elements
                    if 1 < idx < len(token_list) - 4:
                        prefix2 = self.one_hot(self.token_to_string(token_list[idx - 2]))
                        prefix1 = self.one_hot(self.token_to_string(token_list[idx - 1]))
                        suffix1 = self.one_hot(self.token_to_string(token_list[idx + 3]))
                        suffix2 = self.one_hot(self.token_to_string(token_list[idx + 4]))
                    elif 1 == idx < len(token_list) - 4:
                        prefix1 = self.one_hot(self.token_to_string(token_list[idx - 1]))
                        suffix1 = self.one_hot(self.token_to_string(token_list[idx + 3]))
                        suffix2 = self.one_hot(self.token_to_string(token_list[idx + 4]))
                    elif 0 == idx < len(token_list) - 4:
                        suffix1 = self.one_hot(self.token_to_string(token_list[idx + 3]))
                        suffix2 = self.one_hot(self.token_to_string(token_list[idx + 4]))
                    elif 1 < idx == len(token_list) - 4:
                        prefix2 = self.one_hot(self.token_to_string(token_list[idx - 2]))
                        prefix1 = self.one_hot(self.token_to_string(token_list[idx - 1]))
                        # Index offset should be 4 but produces IndexError
                        suffix1 = self.one_hot(self.token_to_string(token_list[idx + 3]))
                    elif 1 < idx:
                        prefix2 = self.one_hot(self.token_to_string(token_list[idx - 2]))
                        prefix1 = self.one_hot(self.token_to_string(token_list[idx - 1]))
                    x = numpy.concatenate([prefix2, prefix1, suffix1, suffix2])
                    y = numpy.concatenate(
                        [self.one_hot(token_str), self.one_hot(next_token_str), self.one_hot(second_next_token_str)])
                    xs.append(x)
                    ys.append(y)
        return xs, ys

    def create_network(self):
        self.net = tflearn.input_data(shape=[None, 1, len(self.string_to_number) * fact_input_dim])
        self.net = tflearn.lstm(self.net, 128, return_seq=True)
        self.net = tflearn.lstm(self.net, 32, return_seq=True)
        self.net = tflearn.lstm(self.net, 32, return_seq=True)
        self.net = tflearn.lstm(self.net, 128)
        self.net = tflearn.fully_connected(self.net, len(self.string_to_number) * 3, activation='softmax')
        self.net = tflearn.regression(self.net)
        self.model = tflearn.DNN(self.net)

    def load(self, token_lists, model_file):
        self.prepare_data(token_lists)
        self.create_network()
        self.model.load(model_file)

    def train(self, token_lists, model_file):
        (xs, ys) = self.prepare_data(token_lists)
        self.create_network()
        xs = numpy.reshape(xs, (-1, 1, len(self.string_to_number) * fact_input_dim))
        self.model.fit(xs, ys, n_epoch=3, batch_size=1024, show_metric=True)
        self.model.save(model_file)

    def query(self, prefix, suffix):
        prefix2 = prefix1 = [0] * len(self.string_to_number)
        if len(prefix) > 1:
            prefix2 = self.one_hot(self.token_to_string(prefix[-2]))
            prefix1 = self.one_hot(self.token_to_string(prefix[-1]))
        elif len(prefix) == 1:
            prefix1 = self.one_hot(self.token_to_string(prefix[-1]))

        suffix1 = suffix2 = [0] * len(self.string_to_number)
        if len(suffix) > 1:
            suffix1 = self.one_hot(self.token_to_string(suffix[0]))
            suffix2 = self.one_hot(self.token_to_string(suffix[1]))
        elif len(suffix) == 1:
            suffix1 = self.one_hot(self.token_to_string(suffix[0]))

        x = numpy.concatenate([prefix2, prefix1, suffix1, suffix2])
        y = self.model.predict(numpy.reshape([x], (-1, 1, len(self.string_to_number) * fact_input_dim)))
        predicted_seq = y[0]
        if type(predicted_seq) is numpy.ndarray:
            predicted_seq = predicted_seq.tolist()

        offset = len(self.string_to_number)
        ps1 = predicted_seq[:offset]
        ps2 = predicted_seq[offset:2 * offset]
        ps3 = predicted_seq[2 * offset:]

        best_number1 = ps1.index(max(ps1))
        best_string1 = self.number_to_string[best_number1]
        best_token1 = self.string_to_token(best_string1)

        res = [best_token1]

        best_number2 = ps2.index(max(ps2))
        best_string2 = self.number_to_string[best_number2]
        if best_string2 != "EOS":
            best_token2 = self.string_to_token(best_string2)
            res.append(best_token2)

        best_number3 = ps3.index(max(ps3))
        best_string3 = self.number_to_string[best_number3]
        if best_string3 != "EOS":
            best_token3 = self.string_to_token(best_string3)
            res.append(best_token3)

        return res

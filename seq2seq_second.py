import GeneticProgramming as GP
from GeneticProgramming.consts import *
from NeuroEncodedExpressionProgramming import build_tree

from keras.models import Model
from keras.layers import Input, SimpleRNN, Dense, Lambda
from keras import backend as K
from keras.utils import to_categorical

import numpy as np

class seq2seq():

    def __init__(self, num_data_encoder_tokens,
                 primitive_set, terminal_set,
                 max_decoder_seq_length=None,
                 timelimit=1,
                 options=None):
        """Initialize the seq2seq model"""

        self.primitive_set = primitive_set
        self.terminal_set = terminal_set
        self.num_data_encoder_tokens = num_data_encoder_tokens
        self.num_samples = 1

        self.target_characters = ['START', 'STOP'] + primitive_set + terminal_set

        get_onehot = lambda index, max_index=len(self.target_characters): np.eye(max_index)[index]

        self.target_token_index = {char: i for i, char in enumerate(self.target_characters)}
        self.target_token_onehot = {char: get_onehot(i) for i, char in enumerate(self.target_characters)}
        self.target_index_token = {i: char for i, char in enumerate(self.target_characters)}

        self.FLoPs = 0

        if options is None:
            self.options = {'use_k-expressions': False}
            self.max_decoder_seq_length = max_decoder_seq_length

        else:
            self.options = options

            self.head_length = self.options['head_length']

            # get max number of children a primitive can have
            n = max([required_children[p] for p in primitive_set])

            self.tail_length = self.head_length*(n-1) + 1
            self.max_decoder_seq_length = self.head_length + self.tail_length

            # We will stop NN form outputing START and STOP
            # We will also stop the NN from outputing anything
            # except terminal when constructing the tail.
            tokens_to_remove = ['START', 'STOP']

            self.start_indices = list(range(len(self.target_token_index)))
            for token in tokens_to_remove:
                self.start_indices.remove(self.target_token_index[token])

            self.start_indices = np.array(self.start_indices)
            
            # get indices of primitive
            tokens_to_remove.extend(self.primitive_set)

            self.primitive_indices = list(range(len(self.target_token_index)))
            for token in tokens_to_remove:
                self.primitive_indices.remove(self.target_token_index[token])

            self.primitive_indices = np.array(self.primitive_indices)

        self.model = self.get_model()

        self.f_hat = lambda x: 0*x[0]

        self.timelimit = timelimit


    def get_model(self):
        """Create the model. This model does not use teacher forcing, meaning that
        the decoder generates is own input data (except for the first token 'START')
        even during training."""

        # +2 for START and STOP
        self.num_decoder_tokens = len(self.target_characters)

        # latent_dim is the dimensionality of the state vector that the encoder/decoder share
        latent_dim = 8

        initial_states = Input((latent_dim,))

        # Define data encoder
        data_encoder_inputs = Input(shape=(None, self.num_data_encoder_tokens))
        data_encoder_rnn1 = SimpleRNN(latent_dim, return_sequences=True, activation='relu')
        data_encoder_rnn2 = SimpleRNN(latent_dim, return_state=True, activation='relu')

        data_encoder_rnn1_output = data_encoder_rnn1(data_encoder_inputs, initial_state=initial_states)
        data_encoder_outputs, data_state_h = data_encoder_rnn2(data_encoder_rnn1_output, initial_state=initial_states)

        # Define equation encoder
        eq_encoder_inputs = Input(shape=(None, len(self.target_characters)))
        eq_encoder_rnn1 = SimpleRNN(latent_dim, return_sequences=True, activation='relu')
        eq_encoder_rnn2 = SimpleRNN(latent_dim, return_state=True, activation='relu')

        eq_encoder_rnn1_output = eq_encoder_rnn1(eq_encoder_inputs, initial_state=initial_states)
        eq_encoder_outputs, eq_state_h = eq_encoder_rnn2(eq_encoder_rnn1_output, initial_state=initial_states)
        
        # We discard `encoder_outputs` and only keep the states.
        # encoder_states = [state_h, state_c]
        encoder_states = K.concatenate((data_state_h, eq_state_h), axis=-1)

        # Set up the decoder, which will only process one timestep at a time.
        decoder_inputs = Input(shape=(1, self.num_decoder_tokens))
        decoder_rnn1 = SimpleRNN(2*latent_dim, return_sequences=True, return_state=True, activation='relu')
        decoder_rnn2 = SimpleRNN(2*latent_dim, return_sequences=True, return_state=True, activation='relu')
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')

        all_outputs = []
        inputs = decoder_inputs
        decoder_rnn1_states = encoder_states

        for _ in range(self.max_decoder_seq_length):

            # Run the decoder on one timestep
            # outputs, state_h, state_c = decoder_lstm(inputs, initial_state=encoder_states)
            decoder_rnn1_outputs, decoder_rnn1_states = decoder_rnn1(inputs, initial_state=decoder_rnn1_states)
            decoder_rnn2_outputs, decoder_rnn2_states = decoder_rnn2(decoder_rnn1_outputs, initial_state=decoder_rnn1_states)

            decoder_dense_outputs = decoder_dense(decoder_rnn2_outputs)
            
            # Store the current prediction (we will concatenate all predictions later)
            all_outputs.append(decoder_dense_outputs)
            
            # Reinject the outputs as inputs for the next loop iteration
            # as well as update the states
            inputs = decoder_dense_outputs

        # Concatenate all predictions
        decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

        # Define and compile model as previously
        model = Model([initial_states, data_encoder_inputs, eq_encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        return model


    def read_decoded_output(self, outputs):
        """Get tokenized output as a string. That is, convert
        the output of the neural network -- softmaxed list of values
        -- to a string of tokens that should represent an equation
        if the neural network has been trained well.
        
        Paramters
        ---------
        outputs : np.array
            The output from the model (decoder).
            The shape is (1, max_decoder_seq_length, num_decoder_tokens).

        Returns
        -------
         : str
            Returns a string that is a lisp without the paraenthesis.
        """
        
        decoded_string_list = ['START']

        for o in outputs[0]:
            i = np.argmax(o)
            token = self.target_index_token[i]

            if token == 'STOP':
                break

            decoded_string_list.append(token)

        return ' '.join(decoded_string_list)


    @staticmethod
    def get_data_encoder_input_data(x, y, f_hat):
        """Given the dataset and the current approximation of the target function,
        get the data to be input into the data encoder.

        Parameters
        ----------
        x : np.array
            The x data. Input data to f_hat. The array is shaped as (num featurs, num input vars).
        y : np.array
            The y data. The desired output of f_hat.
        f_hat : function
            The approimation of the target function.

        Returns
        -------
        encoder_input_data : np.array
            A (1, len(x), 2) shaped array that will be input into the data encoder.
        """

        y_hat = f_hat(x)
        signed_error = y - y_hat

        encoder_input_data = np.array([[[x, e] for x, e in zip(x[0], signed_error)]])

        return encoder_input_data


    def get_eq_encoder_input_data(self, f_hat_seq):
        """Given the the current approximation of the target function as
        a sequence, get the input to the equation encoder.

        Parameters
        ----------
        f_hat_seq : list
            The approimation of the target function as a sequence of strings.

        Returns
        -------
        encoder_input_data : np.array
            A (1, len(f_hat_seq), len(self.target_characters)) shaped array
            that will be input into the equation encoder.
        """

        eq_encoder_input_data = np.array([[self.target_token_onehot[node] for node in f_hat_seq]])

        return eq_encoder_input_data


    def get_decoder_input_data(self):
        """Get the START token for input to the decoder. The other inputs
        to the token will be generated by the previous output of the decoder."""

        # Prepare decoder input data that just contains the start character
        # Note that we could have made it a constant hard-coded in the model
        decoder_input_data = np.zeros((self.num_samples, 1, self.num_decoder_tokens))
        decoder_input_data[:, 0, self.target_token_index['START']] = 1.

        return decoder_input_data


    def evaluate(self, x, y, f_hat, f_hat_seq,
                 return_equation=False,
                 return_equation_str=False,
                 return_decoded_list=False):

        fitness_sum = 0

        for _ in range(self.timelimit):

            output = self.evaluate_single(x, y, f_hat, f_hat_seq,
                                          return_equation=True,
                                          return_decoded_list=True)

            fitness = output['fitness']

            # If the model really produced an equation...
            if output['equation'] is not None:

                f_hat = output['equation']
                f_hat_seq = output['decoded_list']

            fitness_sum += fitness     

        output['fitness'] = fitness_sum
        return output


    def evaluate_single(self, x, y, f_hat, f_hat_seq,
                        initial_states=np.zeros(8)[None, :],
                        return_equation=False,
                        return_equation_str=False,
                        return_decoded_list=False):
        """Evaluate the model"""

        data_encoder_input_data = self.get_data_encoder_input_data(x, y, f_hat)
        eq_encoder_input_data = self.get_eq_encoder_input_data(f_hat_seq)

        decoder_input_data = self.get_decoder_input_data()

        prediction = self.model.predict([initial_states, data_encoder_input_data, eq_encoder_input_data, decoder_input_data])

        if self.options['use_k-expressions']:

            # Don't pick terminals in the tail
            for i, row in enumerate(prediction[0]):
                if i >= self.head_length:
                    prediction[0, i, self.primitive_indices] += 2.
                else:
                    prediction[0, i, self.start_indices] += 2.

        # decoded in terms of seq2seq model -- still a k-expression
        decoded_string = self.read_decoded_output(prediction)

        decoded_list = decoded_string.split(' ')
        
        # If NN has output a STOP, ignore the rest
        # of the output.
        try:
            index = decoded_list.index('STOP')
            decoded_list = decoded_list[:index]

        except ValueError:
            # STOP not in decoded_list, so don't worry about removing it.
            pass

        decoded_list = decoded_list[1:]

        if not self.options['use_k-expressions']:
            # We will adjust this value, if decoded_list
            # actually represents and equation.
            fitness = self.get_penalty(decoded_list,
                                       primitive_set=self.primitive_set,
                                       terminal_set=self.terminal_set)

        if 'START' not in decoded_list:

            if self.options['use_k-expressions']:

                lisp = build_tree(decoded_list)

            else:

                num_terminals = len([x for x in decoded_list if x in self.terminal_set])
                num_primitives = len([x for x in decoded_list if x in self.primitive_set])

                if num_primitives+1 != num_terminals:
                    lisp = None

                else:
                    # This value might be None if decoded_string is not a stripped_lisp
                    lisp = self.get_lisp_from_stripped_lisp(decoded_list, self.primitive_set)

            if lisp is not None:
                if self.is_equation(lisp):
                    t = GP.Individual(rng=None, primitive_set=self.primitive_set, terminal_set=self.terminal_set,
                                      tree=lisp, actual_lisp=True)
                    t.evaluate_fitness([np.vstack((y,x)).T], compute_val_error=False)
                    fitness = t.fitness[0]

        output = {'fitness': fitness}

        if return_equation:
            output['equation'] = self.f_hat if fitness < 10**9 else None

        if return_equation_str:
            output['equation_str'] = lisp if fitness < 10**9 else None

        if return_decoded_list:
            output['decoded_list'] = decoded_list if fitness < 10**9 else None

        output['raw_decoded_list'] = decoded_list

        return output


    @staticmethod
    def get_penalty(decoded_list, primitive_set, terminal_set):
        """Get a penalty for non-equation that will
        guide the NN to produces equation in the future.

        In a binary tree, there is one more leaf than internal
        node. Thus, we penalize the supposed equation based on
        how far it is from this equality.
        
        Parameters
        ---------
        decoded_list : list of str
            The equation
        """
        
        # Assume not an equation, we will
        # overwrite the penalty later if it
        # turns out that this is an equation.
        penalty = 10**9

        num_terminals = len([x for x in decoded_list if x in terminal_set])
        num_primitives = len([x for x in decoded_list if x in primitive_set])

        penalty += 10**9*np.abs(num_primitives+1-num_terminals)/(len(decoded_list)+1)

        if num_primitives == 0 and num_terminals == 0:
            penalty += (10**9)*100    # should get the max possible length of decoded_list rather than 100

        num_starts = len([x for x in decoded_list if x == 'START'])

        penalty += 10**9*num_starts

        if num_terminals + num_primitives % 2 == 0:
            penalty += 10**9

        return penalty


    def is_equation(self, eq):
        """Check if eq is an equation

        Parameters
        ----------
        eq : str
            A lisp (s-expression, infix notation) of an equation?

        Returns
        -------
         : bool
            If True, eq is an equation in the form of a lisp.
        """

        try:
            t = GP.Tree(rng=None, primitive_set=self.primitive_set, terminal_set=self.terminal_set,
                        tree=eq, actual_lisp=True)
            f = eval('lambda x:' + t.convert_lisp_to_standard_for_function_creation())
            f([1])
            self.f_hat = f
            return True

        except SyntaxError:
            return False


    @staticmethod
    def get_lisp_from_stripped_lisp(stripped_lisp, primitives):
        """Given a lisp (also known as an s-expression) without the
        parenthesis, put the parenthesis back.

        Parameters
        ----------
        stripped_lisp : str or list
            If str, there must be spaces between each primtive/terminal. If
            list, it must be a list of strings where the elements are the 
            primitives/terminals in order.
        primitives : list
            A list of strings containing all the primitives that are allowed
            to be used in the equation.

        Returns
        -------
        lisp : str
            The lisp with parenthesis. If input cannot be a stripped_lisp
            return None.
        """

        if type(stripped_lisp) == str:
            stripped_lisp = stripped_lisp.split(' ')

        if len(stripped_lisp) == 0:
            return None

        lisp_list = []
        index = []

        # get locations of the primitives
        for i, char in enumerate(stripped_lisp):
            
            if char not in primitives:
                lisp_list.append(char)

            else:
                index.append(i)
                lisp_list.append('('+char)

        # put in all '(' and ')'
        for i in reversed(index):

            # figure out where to put )
            children = 2
            j = 1
            while j < children or i+j in index:
                if i+j in index:
                    children += 2
                j += 1

            # If the input to this function
            # was not a stripped list, return None.
            if i+j >= len(lisp_list):
                return None

            lisp_list[i+j] += ')'

        lisp = ' '.join(lisp_list)

        # If the input to this function
        # was not a stripped list, return None.
        if lisp[0] == '(' and lisp[-1] == ')':
            return lisp

        else:
            return None


    @staticmethod
    def set_weights(weights, model):

        weight_shapes = [(0,0)]
        start = 0

        for layer in model.layers:

            layer_weights = layer.get_weights()

            if len(layer_weights) == 2:

                weight_shapes.append(layer_weights[0].shape)

                end = start + np.prod(weight_shapes[-1])

                new_weights = [weights[start:end].reshape(weight_shapes[-1]),
                               np.zeros(layer_weights[-1].shape)]

                layer.set_weights(new_weights)
                start = end

            elif len(layer_weights) == 3:

                weight_shapes.extend([layer_weights[0].shape, layer_weights[1].shape])

                end1 = start + np.prod(weight_shapes[-2])
                end2 = end1 + np.prod(weight_shapes[-1])

                new_weights = [weights[start:end1].reshape(weight_shapes[-2]),
                               weights[end1:end2].reshape(weight_shapes[-1]),
                               np.zeros(layer_weights[-1].shape)]
                layer.set_weights(new_weights)
                start = end2


    @staticmethod
    def get_num_weights(model):

        # get number of weights
        num_weights = 0
        for layer in model.layers:
            layer_weights = layer.get_weights()

            # Things like input layer have length 0.
            if len(layer_weights) > 0:
                num_weights += np.prod(layer_weights[0].shape)

                if len(layer_weights) == 3:
                    num_weights += np.prod(layer_weights[1].shape)

        return num_weights


if __name__ == '__main__':

    num_samples = 1

    # options = None
    options = {'use_k-expressions': True,
               'head_length': 3}    

    s2s = seq2seq(num_data_encoder_tokens=2,
                  primitive_set=['*', '+', '-'],
                  terminal_set=['x0'],
                  max_decoder_seq_length=30,
                  timelimit=10,
                  options=options)

    x = np.linspace(-1, 1, 20)[None, :]
    f = lambda x: x[0]**2
    y = f(x)
    f_hat = lambda x: 0*x[0]
    # f_hat_seq = ['START', '-', 'x0', 'x0', 'STOP']
    f_hat_seq = ['START', '-', 'x0', 'x0']# + ['x0']*(len()-4)

    # import pandas as pd
    # import itertools
    # # from keras.models import load_model
    # # s2s.model = load_model('/Users/rgrindle/Documents/model_saving_test.h5')
    # def save_model_weights(model):

    #     weights = model.get_weights()

    #     weight_shapes = [w.shape for w in weights]
    #     flattened_weights = list(itertools.chain(*[w.flatten() for w in weights]))

    #     pd.DataFrame(flattened_weights).to_csv('/Users/rgrindle/Documents/model_weights_saving_test.csv', index=False, header=None)


    # def load_model_weights(model):

    #     weights = model.get_weights()
    #     weight_shapes = [w.shape for w in weights]

    #     loaded_weights = pd.read_csv('/Users/rgrindle/Documents/model_weights_saving_test.csv', header=None).iloc[:, :].values

    #     reshaped_loaded_weights = []
    #     start = 0

    #     for shape in weight_shapes:

    #         length = np.prod(shape)

    #         reshaped_loaded_weights.append(loaded_weights[start:start+length].reshape(shape))

    #         start += length

    #     model.set_weights(reshaped_loaded_weights)

    #     return reshaped_loaded_weights


    # # save_model_weights(s2s.model)
    # weights = load_model_weights(s2s.model)

    # stuff = sum([np.sum(w1 - w2) for w1, w2 in zip(weights, s2s.model.get_weights())])
    # # stuff = [sum(w) if type(w) == np.ndarray else w for w in stuff]
    # print('stuff', stuff)

    output = s2s.evaluate(x, y, f_hat, f_hat_seq,
                          return_decoded_list=True)
    print('fitness', output['fitness'])
    print('final nn output', output['raw_decoded_list'])

    # save the model
    s2s.model.save('/Users/rgrindle/Documents/model_saving_test.h5')

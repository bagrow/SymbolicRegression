import GeneticProgramming as GP

from keras.models import Model
from keras.layers import Input, SimpleRNN, Dense, Lambda
from keras import backend as K

import numpy as np

class seq2seq():

    def __init__(self, num_encoder_tokens, primitive_set, terminal_set, max_decoder_seq_length):
        """Initialize the seq2seq model"""

        self.primitive_set = primitive_set
        self.terminal_set = terminal_set
        self.num_encoder_tokens = num_encoder_tokens
        self.max_decoder_seq_length = max_decoder_seq_length
        self.num_samples = 1

        self.target_characters = ['START', 'STOP'] + primitive_set + terminal_set

        self.target_token_index = {char: i for i, char in enumerate(self.target_characters)}
        self.target_index_token = {i: char for i, char in enumerate(self.target_characters)}

        self.FLoPs = 0

        self.model = self.get_model()

        self.f_hat = lambda x: 0*x[0]


    def get_model(self):
        """Create the model. This model does not use teacher forcing, meaning that
        the decoder generates is own input data (except for the first token 'START')
        even during training."""

        # +2 for START and STOP
        self.num_decoder_tokens = len(self.target_characters)

        # latent_dim is the dimensionality of the state vector that the encoder/decoder share
        latent_dim = 8

        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, self.num_encoder_tokens))
        encoder_rnn1 = SimpleRNN(latent_dim, return_sequences=True, activation='relu')
        encoder = SimpleRNN(latent_dim, return_state=True, activation='relu')
        # encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # encoder_outputs, state_h = encoder(encoder_inputs)
        rnn1_output = encoder_rnn1(encoder_inputs)
        encoder_outputs, state_h = encoder(rnn1_output)
        
        # We discard `encoder_outputs` and only keep the states.
        # encoder_states = [state_h, state_c]
        encoder_states = state_h


        # Set up the decoder, which will only process one timestep at a time.
        decoder_inputs = Input(shape=(1, self.num_decoder_tokens))
        decoder_rnn1 = SimpleRNN(latent_dim, return_sequences=True, return_state=True, activation='relu')
        decoder_rnn2 = SimpleRNN(latent_dim, return_sequences=True, return_state=True, activation='relu')
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')

        all_outputs = []
        inputs = decoder_inputs
        rnn1_states = encoder_states

        for _ in range(self.max_decoder_seq_length):

            # Run the decoder on one timestep
            # outputs, state_h, state_c = decoder_lstm(inputs, initial_state=encoder_states)
            rnn1_outputs, rnn1_states = decoder_rnn1(inputs, initial_state=rnn1_states)
            rnn2_outputs, rnn2_states = decoder_rnn2(rnn1_outputs)

            dense_outputs = decoder_dense(rnn2_outputs)
            
            # Store the current prediction (we will concatenate all predictions later)
            all_outputs.append(dense_outputs)
            
            # Reinject the outputs as inputs for the next loop iteration
            # as well as update the states
            inputs = dense_outputs
            # states = [state_h, state_c]
            # states = state_h


        # Concatenate all predictions
        decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

        # Define and compile model as previously
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
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
    def get_encoder_input_data(x, y, f_hat):
        """Given the dataset and the current approximation of the target function,
        get the data to be input into the encoder.

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
            A (1, len(x), 2) shaped array that is will be input into the encoder.
        """

        y_hat = f_hat(x)
        signed_error = y - y_hat

        encoder_input_data = np.array([[[x, e] for x, e in zip(x[0], signed_error)]])

        return encoder_input_data


    def get_decoder_input_data(self):
        """Get the START token for input to the decoder. The other inputs
        to the token will be generated by the previous output of the decoder."""

        # Prepare decoder input data that just contains the start character
        # Note that we could have made it a constant hard-coded in the model
        decoder_input_data = np.zeros((self.num_samples, 1, self.num_decoder_tokens))
        decoder_input_data[:, 0, self.target_token_index['START']] = 1.

        return decoder_input_data


    def evaluate(self, x, y, f_hat, return_equation=False, return_equation_str=False):
        """Evaluate the model"""

        encoder_input_data = self.get_encoder_input_data(x, y, f_hat)

        decoder_input_data = self.get_decoder_input_data()

        prediction = self.model.predict([encoder_input_data, decoder_input_data])

        decoded_string = self.read_decoded_output(prediction)

        decoded_list = decoded_string.split(' ')
        
        # If NN has out a STOP, ignore the rest
        # of the output.
        try:
            index = decoded_list.index('STOP')
            decoded_list = decoded_list[:index]

        except ValueError:
            # STOP not in decoded_list, so don't worry about removing it.
            pass

        decoded_list = decoded_list[1:]

        # We will adjust this value, if decoded_list
        # actually represents and equation.
        fitness = self.get_penalty(decoded_list,
                                   primitive_set=self.primitive_set,
                                   terminal_set=self.terminal_set)

        if 'START' not in decoded_list:

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

        if return_equation:
            if fitness < 10**9:
                return fitness, self.f_hat

            else:
                return fitness, None

        elif return_equation_str:
            if fitness < 10**9:
                return fitness, lisp

            else:
                return fitness, None

        else:
            return fitness


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
            print('0 == 0')

        num_starts = len([x for x in decoded_list if x == 'START'])

        penalty += 10**9*num_starts

        if num_terminals + num_primitives % 2 == 0:
            penalty += 10**9
            print('even', penalty, decoded_list)

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
            print('eq', eq)
            print('lisp', t.get_lisp_string())
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

    s2s = seq2seq(num_encoder_tokens=2,
                  primitive_set=['*', '+'],
                  terminal_set=['x0'],
                  max_decoder_seq_length=30)

    x = np.linspace(-1, 1, 20)[None, :]
    f = lambda x: x[0]**2
    y = f(x)
    f_hat = lambda x: 0*x[0]

    fitness = s2s.evaluate(x, y, f_hat)
    print('fitness', fitness)

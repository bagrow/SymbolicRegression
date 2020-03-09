import GeneticProgramming as GP
from GeneticProgramming.consts import *
from kexpressions import build_tree
from GeneticProgramming.protected_functions import *

from keras.models import Model
from keras.layers import Input, SimpleRNN, Dense, Lambda
from keras import backend as K
from keras.utils import to_categorical

import numpy as np
import pandas as pd

import itertools
import copy

class TlcsrNetwork():

    def __init__(self, rng, num_data_encoder_inputs,
                 primitive_set, terminal_set,
                 max_decoder_seq_length=None,
                 timelimit=1, use_constants=False,
                 options=None):
        """Initialize the TLC-SR network

        Parameters
        ----------
        rng : np.random.RandomState
            Use this for reproducible results
        num_data_encoder_inputs : int
            The number of input nodes to
            the encoder will encounter. So far,
            we have only set this to 2 for x-values
            and error at those x-values.
        primitive_set : list
            The allowed primitives.
        terminal_set : list
            The allowed terminals. For constants, only
            put specific constants in this list. If general
            constants are desired use use_constants=True.
        max_decoder_seq_length : int (default=None)
            If using k-expression set head_length in
            options to an integer then max_decoder_seq_length
            will be computed. Otherwise, specify this yourself.
            This number refers to the maximum length output the
            decoder is allowed to produce.
        timelimit : int
            The maximum number of rewrites the network is allowed.
        use_constants : bool
            If true, two nodes will be added to the output of data
            encoder: one to let the NN choose to place a constant
            (part of one-hot vector) and another that specifies the
            of the constant to use (not part of softmax).
        options : dict
            Some additional options include:
            'use_k-expressions' : bool
                Set to true if you want equations interpreted as k-expression.
            'head_length' : int
                Length of head in k-expression. Only use
                if use_k-expressions=True.
        """

        self.rng = rng

        self.use_constants = use_constants

        self.primitive_set = primitive_set
        self.terminal_set = terminal_set

        if self.use_constants:
            assert '#f' in self.terminal_set, '#f must be in terminal set when using constants'

        self.num_data_encoder_inputs = num_data_encoder_inputs
        self.num_samples = 1

        self.target_characters = ['START', 'STOP'] + primitive_set + terminal_set

        if self.use_constants:
            get_onehot = lambda index, max_index=len(self.target_characters)+1: np.eye(max_index)[index]


        else:
            get_onehot = lambda index, max_index=len(self.target_characters): np.eye(max_index)[index]

        # Create dictionaries to map between tokens indices and one-hot vectors of tokens
        self.target_token_index = {char: i for i, char in enumerate(self.target_characters)}
        self.target_token_onehot = {char: get_onehot(i) for i, char in enumerate(self.target_characters)}
        self.target_index_token = {i: char for i, char in enumerate(self.target_characters)}

        self.effort = 0

        if options is None:

            # Default to s-expressions.
            self.options = {'use_k-expressions': False}
            self.max_decoder_seq_length = max_decoder_seq_length

        else:
            self.options = options

            # Get lenght of decoder output based
            # on head length.
            self.head_length = self.options['head_length']

            # get max number of children a primitive can have
            n = max([required_children[p] for p in primitive_set])

            self.tail_length = self.head_length*(n-1) + 1
            self.max_decoder_seq_length = self.head_length + self.tail_length

            # We will stop NN form outputing START and STOP
            # We will also stop the NN from outputing anything
            # except terminal when constructing the tail.
            tokens_to_remove = ['START', 'STOP']

            self.not_start_indices = list(range(len(self.target_token_index)))
            for token in tokens_to_remove:
                self.not_start_indices.remove(self.target_token_index[token])

            self.not_start_indices = np.array(self.not_start_indices)
            
            # Get indices of primitive. Will be used when enforcing
            # terminal output in tail.
            tokens_to_remove.extend(self.primitive_set)

            self.terminal_indices = list(range(len(self.target_token_index)))
            for token in tokens_to_remove:
                self.terminal_indices.remove(self.target_token_index[token])

            self.terminal_indices = np.array(self.terminal_indices)

        if 'eq_encoder_only' not in self.options:
            self.options['eq_encoder_only'] = False

        if 'data_encoder_only' not in self.options:
            self.options['data_encoder_only'] = False

        # construct the neural network
        self.network = self.get_network(data_encoder_only=self.options['data_encoder_only'],
                                        eq_encoder_only=self.options['eq_encoder_only'])

        self.timelimit = timelimit


    def get_network(self, data_encoder_only=False, eq_encoder_only=False):
        """Create the network. This network does not use
        teacher forcing, meaning that the decoder generates
        is own input data (except for the first token 'START')
        even during training.

        The network has a decoder and two encoders: equation encoder
        and data encoder. The equation encoder takes one-hot vector
        representations of equations. The data encoder takes data
        related to the current equation at each x-value as a sequence.
        The decoder outputs a one-hot vector representation of a new
        equation.

        Parameters
        ----------
        data_encoder_only : bool
            If true, load entire network with outputs from
            data_encoder only. Entire network
            is kept for loading weights. TODO: Might need to change
            this so you don't have to keep all weights since it will
            not be helpful in some situations.
        eq_encoder_only : bool
            If true, load entire network with additional outputs
            from the eq_encoder only. Entire network
            is kept for loading weights. TODO: Might need to change
            this so you don't have to keep all weights since it will
            not be helpful in some situations.
        """

        assert not (data_encoder_only and eq_encoder_only), 'Both data_encoder_only and eq_encoder_only cannot be used at the same time'

        # This excludes constant value node if used.
        self.num_decoder_tokens = len(self.target_characters)

        # latent_dim is the dimensionality of the
        # state vector that the encoders share.
        # The state vector of the decoder sill
        # have 2*laten_dim to be able to fit
        # the state vectors from both encoders.
        latent_dim = 8

        # These are the initial hidden values of
        # both encoders. This is done for reproducibility.
        initial_states = Input((latent_dim,))

        # Define data encoder
        data_encoder_inputs = Input(shape=(None, self.num_data_encoder_inputs))
        data_encoder_rnn1 = SimpleRNN(latent_dim, return_state=True, return_sequences=True, activation='relu')
        data_encoder_rnn2 = SimpleRNN(latent_dim, return_state=True, activation='relu')

        data_encoder_rnn1_output, data_state_h1 = data_encoder_rnn1(data_encoder_inputs, initial_state=initial_states)
        data_encoder_outputs, data_state_h2 = data_encoder_rnn2(data_encoder_rnn1_output, initial_state=initial_states)

        # Define equation encoder
        if self.use_constants:
            eq_encoder_inputs = Input(shape=(None, len(self.target_characters)+1))
        else:
            eq_encoder_inputs = Input(shape=(None, len(self.target_characters)))
        eq_encoder_rnn1 = SimpleRNN(latent_dim, return_state=True, return_sequences=True, activation='relu')
        eq_encoder_rnn2 = SimpleRNN(latent_dim, return_state=True, activation='relu')

        eq_encoder_rnn1_output, eq_state_h1 = eq_encoder_rnn1(eq_encoder_inputs, initial_state=initial_states)
        eq_encoder_outputs, eq_state_h2 = eq_encoder_rnn2(eq_encoder_rnn1_output, initial_state=initial_states)
        
        # We discard `encoder_outputs` and only keep the states.
        # Before putting these into the decoder, we must concatenate.
        # The Lambda is necessary here for reproducible results. I don't
        # understand why.
        encoder_states_layer1 = Lambda(lambda cat_list: K.concatenate((cat_list[0], cat_list[1]), axis=1))([data_state_h1, eq_state_h1])
        encoder_states_layer2 = Lambda(lambda cat_list: K.concatenate((cat_list[0], cat_list[1]), axis=1))([data_state_h2, eq_state_h2])

        # Set up the decoder, which will only process one timestep at a time
        # because it will take its previous output as input.
        if self.use_constants:
            decoder_inputs = Input(shape=(1, self.num_decoder_tokens+1))

        else:
            decoder_inputs = Input(shape=(1, self.num_decoder_tokens))
        decoder_rnn1 = SimpleRNN(2*latent_dim, return_sequences=True, return_state=True, activation='relu', name='decoder1')
        decoder_rnn2 = SimpleRNN(2*latent_dim, return_sequences=True, return_state=True, activation='relu', name='decoder2')

        if self.use_constants:
            # separate constant value from other, so that 
            # the activation function can be different
            decoder_dense_const = Dense(1, activation='tanh', name='decoder_const_out')
            decoder_dense = Dense(self.num_decoder_tokens, activation='softmax', name='decoder_dense')

        else:
            decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')

        all_outputs = []
        all_outputs_const = []
        inputs = decoder_inputs
        decoder_rnn1_states = encoder_states_layer1
        decoder_rnn2_states = encoder_states_layer2

        for _ in range(self.max_decoder_seq_length):

            # Run the decoder on one timestep
            decoder_rnn1_outputs, decoder_rnn1_states = decoder_rnn1(inputs, initial_state=decoder_rnn1_states)
            decoder_rnn2_outputs, decoder_rnn2_states = decoder_rnn2(decoder_rnn1_outputs, initial_state=decoder_rnn2_states)

            decoder_dense_outputs = decoder_dense(decoder_rnn2_outputs)
            
            if self.use_constants:
                decoder_dense_const_output = decoder_dense_const(decoder_rnn2_outputs)
            
            # Store the current prediction (we will concatenate all predictions later)
            all_outputs.append(decoder_dense_outputs)
            
            if self.use_constants:
                all_outputs_const.append(decoder_dense_const_output)
            
            # Reinject the outputs as inputs for the next loop iteration
            # as well as update the states
            if self.use_constants:
                inputs = Lambda(lambda x: K.concatenate(x))([decoder_dense_outputs, decoder_dense_const_output])

            else:
                inputs = decoder_dense_outputs

        # Concatenate all predictions
        decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

        if self.use_constants:
            decoder_outputs_const = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs_const)

        # Define and compile newtork
        if self.use_constants:
            input_list = [initial_states, data_encoder_inputs, eq_encoder_inputs, decoder_inputs]
            output_list = [decoder_outputs, decoder_outputs_const]
    
        else:
            input_list = [initial_states, data_encoder_inputs, eq_encoder_inputs, decoder_inputs]
            output_list = [decoder_outputs]

        if data_encoder_only:
            output_list.extend([data_state_h1, data_state_h2, data_encoder_outputs])
        elif eq_encoder_only:
            output_list.extend([eq_state_h1, eq_state_h2, eq_encoder_outputs])
            
        network = Model(input_list, output_list)
        network.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        # calculate effort of evaluating NN based on
        # length of input to equation encoder and data encoder
        # Save as function for when we know these value
        eq_nodes = len(self.terminal_set)+len(self.primitive_set)

        if self.use_constants:
            eq_nodes += 1   # for constant-value node

        output_len = self.head_length+self.tail_length
        activations_in_network = lambda eq_input_len, data_input_len: eq_input_len*(eq_nodes+latent_dim*2) + data_input_len*(self.num_data_encoder_inputs +  latent_dim*2) + output_len*((2*latent_dim)*2 + eq_nodes)

        # for single layer with n nodes and previous layer of m nodes,
        # there are m-1 additions for each of the n nodes. Thus, (m-1)*n
        additions_in_network = lambda eq_input_len, data_input_len: eq_input_len*((eq_nodes-1)*latent_dim + 3*latent_dim*(latent_dim-1)) + data_input_len*((self.num_data_encoder_inputs-1)*latent_dim + 3*latent_dim*(latent_dim-1)) + (3*(2*latent_dim-1)*(2*latent_dim) + (2*latent_dim-1)*eq_nodes)*output_len

        weights_in_eq_encoder = lambda eq_input_len: eq_input_len*(eq_nodes*latent_dim + 3*latent_dim**2) 
        weights_in_data_encoder = lambda data_input_len: data_input_len*(self.num_data_encoder_inputs*latent_dim + 3*latent_dim**2)
        weights_in_decoder = (output_len)*(3*(2*latent_dim)**2 + (2*latent_dim)*eq_nodes)
 
        weights_per_eval = lambda eq_input_len, data_input_len: weights_in_eq_encoder(eq_input_len) + weights_in_data_encoder(data_input_len) + weights_in_decoder

        self.effort_in_eval = lambda eq_input_len, data_input_len: weights_per_eval(eq_input_len, data_input_len) + activations_in_network(eq_input_len, data_input_len) + additions_in_network(eq_input_len, data_input_len)

        return network


    def read_decoded_output(self, outputs, const_outputs=None):
        """Get tokenized output as a string. That is, convert
        the output of the neural network -- softmaxed list of values
        -- to a string of tokens that should represent an equation
        if the neural network has been trained well or if using
        k-expressions.
        
        Paramters
        ---------
        outputs : np.array
            The output from the decoder.
            The shape is (1, max_decoder_seq_length, num_decoder_tokens).

        const_outputs : list (default=None)
            Used only if using constants. If so, this is a list of
            the output constant values of shape (1, max_decoder_seq_length, 1).

        Returns
        -------
         : str
            Returns a string that is a list of tokens. Either
            an s-expression (no parens) or k-expression.
        """

        decoded_string_list = ['START']

        if self.use_constants:

            for o, c in zip(outputs[0], const_outputs[0]):
                i = np.argmax(o)
                token = self.target_index_token[i]

                if token == '#f':
                    token = str(c[0])

                if token == 'STOP':
                    break

                decoded_string_list.append(token)

        else:

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
        x : np.array (num observations by num input vars)
            The x data. Input data to f_hat.
            The array is shaped as (num featurs, num input vars).
        y : np.array (num observations)
            The y data. The desired output of f_hat.
        f_hat : function
            The approximation of the target function.

        Returns
        -------
        encoder_input_data : np.array
            A (1, num input vars, 2) shaped array that will be
            input into the data encoder.
        """

        y_hat = f_hat(x.T)
        signed_error = y - y_hat

        encoder_input_data = np.array([[[*x, e] for x, e in zip(x, signed_error)]])

        return encoder_input_data


    def get_eq_encoder_input_data(self, f_hat_seq):
        """Given the the current approximation of the target function as
        a sequence, get the input to the equation encoder.

        Parameters
        ----------
        f_hat_seq : list
            The approximation of the target function as a sequence of strings.

        Returns
        -------
        encoder_input_data : np.array
            A (1, len(f_hat_seq), len(self.target_characters)) shaped array
            that will be input into the equation encoder.
        """

        eq_encoder_input_data = []

        for node in f_hat_seq:

            if node in self.target_token_onehot:
                eq_encoder_input_data.append(self.target_token_onehot[node])

            else:   # node is a string of a number(s)
                input_vec = copy.copy(self.target_token_onehot['#f'])
                # index = self.target_token_index['constant_value']
                input_vec[-1] = float(node)
                eq_encoder_input_data.append(input_vec)

        return np.array([eq_encoder_input_data])


    def get_decoder_input_data(self):
        """Get the START token for input to the decoder. The other inputs
        to the token will be generated by the previous output of the decoder.
        TODO: This could probably to hard-coded into the network.

        Returns
        -------
        decoder_input_data : np.array (shape=(num observations, 1, one-hot vector length))
        """

        # Prepare decoder input data that just contains the start character.

        if self.use_constants:
            decoder_input_data = np.zeros((self.num_samples, 1, self.num_decoder_tokens+1))

        else:
            decoder_input_data = np.zeros((self.num_samples, 1, self.num_decoder_tokens))

        decoder_input_data[:, 0, self.target_token_index['START']] = 1.

        return decoder_input_data


    def evaluate(self, x, y, f_hat, f_hat_seq,
                 return_equation=False,
                 return_equation_str=False,
                 return_decoded_list=False,
                 return_errors=False):
        """Get lowest error of TLC-SR (multiple rewrites)
        on a particular dataset.

        Parameters
        ----------
        x : np.array
            x-data from dataset
        y : np.array
            y-data from dataset
        f_hat : function
            The initial approximation of dataset.
        f_hat_seq : list
            The initial string representation of f_hat.
        return_equation : bool
            If true, return the lowest error function.
        return_equation_str : bool
            If true, return the string representation of
            the lowest error function.
        return_decoded_list : bool
            If true, return the string representation of
            most recent output equation.
        return_errors : bool
            If true, return all errors.

        Return
        ------
        output : dict
            Included info depends on return_...
            parameters.
        """

        error_sum = 0
        errors = []

        # Get the lowest error.
        error_best = float('inf')

        # We will keep track of the min error
        # over the previous period scores.
        # TODO: pass to class as parameter
        period = 5

        equation_strs = []
        self.summary_data = []

        # TODO: save each equation and each error
        # Important to start at t=1, so that the
        # condition for adding group score works.
        # Otherwise, will at when t=0...
        for t in range(1, 1+self.timelimit):

            output = self.rewrite_equation(x, y, f_hat, f_hat_seq,
                                           return_equation=True,
                                           return_decoded_list=True,
                                           return_equation_str=True)

            error = output['error']

            if error < error_best:
                error_best = error
                equation_best = output['equation_str']

            errors.append(error)
            equation_strs.append(output['equation_str'])

            # If the network really produced an equation...
            # This check is unnecessary for k-expressions.
            if output['equation'] is not None:

                f_hat = output['equation']
                f_hat_seq = output['decoded_list']

            error_sum += error

            # If solution has deteriorated between past
            # two groups, stop rewriting equations.
            if len(errors) >= 2*period:
                older_group_min = min(errors[-2*period:-period])
                newer_group_min = min(errors[-period:])

                if older_group_min <= newer_group_min:
                    break

        if not return_equation:
            del output['equation']

        if not return_decoded_list:
            del output['decoded_list']
            del output['raw_decoded_list']

        if return_equation_str:
            output['equation_str'] = equation_strs

        if return_errors:
            output['errors'] = errors

        output['error_sum'] = error_sum
        output['error_best'] = error_best
        output['equation_best'] = equation_best

        return output


    def get_network_output(self,
                           initial_states,
                           data_encoder_input_data,
                           eq_encoder_input_data,
                           decoder_input_data):

        return self.network.predict([initial_states, data_encoder_input_data, eq_encoder_input_data, decoder_input_data])


    def rewrite_equation(self, x, y, f_hat, f_hat_seq,
                         initial_states=np.zeros(8)[None, :],
                         return_equation=False,
                         return_equation_str=False,
                         return_decoded_list=False):
        """Rewrite the equation using the network.

        Parameters
        ----------
        x : np.array
            The input data for the dataset.
        y : np.array
            The output data for the dataset.
        f_hat : function
            The current approximation of the dataset.
        f_hat_seq : list
            List of strings. Each string is a token
            in the equation.
        initial_states : np.array (default all zeros)
            The initial states of the encoders hidden
            values.
        return_equation : bool
            If true, return the function (which does equation)
            output by the netork
        return_equation_str : bool
            If true, return the string representation
            of the equation output by network.
        return_decoded_list : bool
            If true, return the output of the network after
            it has been converted to tokens.

        Returns
        -------
        output : dict
            The keys depend on the return_... parameters.
        """

        # Get input ready.
        data_encoder_input_data = self.get_data_encoder_input_data(x, y, f_hat)
        eq_encoder_input_data = self.get_eq_encoder_input_data(f_hat_seq)
        decoder_input_data = self.get_decoder_input_data()

        all_network_outputs = self.get_network_output(initial_states,
                                                      data_encoder_input_data,
                                                      eq_encoder_input_data,
                                                      decoder_input_data)

        # decoder output
        if len(all_network_outputs) == 1:
            prediction = all_network_outputs
        else:
            prediction = all_network_outputs[0]

        if self.use_constants:
            constant_value = all_network_outputs[1]

        extra_outputs = all_network_outputs[2:]
        output = {key: value for key, value in zip(['state_h1', 'state_h2', 'encoder_output'], extra_outputs)}

        self.effort += self.effort_in_eval(eq_input_len=len(eq_encoder_input_data[0]), 
                                            data_input_len=len(data_encoder_input_data[0]))

        if self.options['use_k-expressions']:

            for i, row in enumerate(prediction[0]):
                if i >= self.head_length:
                    prediction[0, i, self.terminal_indices] += 2.
                else:
                    prediction[0, i, self.not_start_indices] += 2.

        # decoded in terms of seq2seq network -- still a k-expression
        if self.use_constants:
            decoded_string = self.read_decoded_output(outputs=prediction,
                                                      const_outputs=constant_value)

        else:
            decoded_string = self.read_decoded_output(outputs=prediction)

        decoded_list = decoded_string.split(' ')
        
        # If NN has output a STOP, ignore the rest
        # of the output.
        try:
            index = decoded_list.index('STOP')
            decoded_list = decoded_list[:index]

        except ValueError:
            # STOP not in decoded_list, so don't worry about removing it.
            pass

        # Remove START token
        decoded_list = decoded_list[1:]

        if not self.options['use_k-expressions']:
            # We will adjust this value, if decoded_list
            # actually represents an equation.
            error = self.get_penalty(decoded_list,
                                     primitive_set=self.primitive_set,
                                     terminal_set=self.terminal_set)

        # nan's can appear in the output of the network
        # if inf's are subtracted. inf's can appear when
        # weights are too large, which is easier to do
        # with reucurrance.
        if np.any(np.isnan(prediction)):
            error = float('inf')

        # if START is in decoded list, keep the penalty already computed
        # otherwise get the actual error
        elif 'START' not in decoded_list:

            if self.options['use_k-expressions']:

                lisp, short_gene = build_tree(decoded_list, return_short_gene=True)
                self.summary_data.append(self.get_lisp_summary(lisp, self.primitive_set, self.terminal_set))

            else:

                num_terminals = len([x for x in decoded_list if x in self.terminal_set])
                num_primitives = len([x for x in decoded_list if x in self.primitive_set])

                if num_primitives+1 != num_terminals:
                    lisp = None

                else:
                    # This value might be None if decoded_string is not a
                    # stripped_lisp
                    lisp = self.get_lisp_from_stripped_lisp(decoded_list, self.primitive_set)

            if lisp is not None:
                if self.is_equation(lisp):

                    t = GP.Individual(rng=None, primitive_set=self.primitive_set, terminal_set=self.terminal_set,
                                      tree=lisp, actual_lisp=True)

                    dataset = [np.vstack((y,x[:,0])).T, []]
                    t.evaluate_fitness(dataset, compute_val_error=False)
                    error = t.fitness[0]
                    self.f_hat = t.f
                    self.effort += t.get_effort_tree_eval(dataset)

                else:
                    print('ERROR: lisp is not an equation')
                    print('lisp =', lisp)
                    exit()

            else:
                print('ERROR: lisp is None')
                print('decoded_list =', decoded_list)
                exit()
        else:
            print('ERROR: START is in decoded_list')
            print('decoded_list =', decoded_list)
            exit()

        output['error'] = error

        if return_equation:
            # This check is if an invalid equation was generated.
            output['equation'] = self.f_hat if error < 10**9 else None

        if return_equation_str:
            # This check is if an invalid equation was generated.
            output['equation_str'] = lisp if error < 10**9 else None

        if return_decoded_list:
            # This check is if an invalid equation was generated.
            output['decoded_list'] = decoded_list if error < 10**9 else None

        output['raw_decoded_list'] = decoded_list

        return output


    @staticmethod
    def get_penalty(decoded_list, primitive_set, terminal_set):
        """Get a penalty for non-equation that will
        guide the NN to produce equation in the future.

        In a binary tree, there is one more leaf than internal
        node. Thus, we penalize the supposed equation based on
        how far it is from this equality.
        
        Parameters
        ---------
        decoded_list : list of str
            The equation as a list of tokens

        Returns
        -------
        penalty : float
            Multiple of 10**9
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

        for element in eq.split(' '):

            element = element.replace('(', '').replace(')', '')

            constant = False

            if '#f' in self.terminal_set:
                try:
                    float(element)
                    constant = True
                except ValueError:
                    pass

            if constant:
                pass
            elif element not in self.primitive_set and element not in self.terminal_set:
                return False

        try:
            t = GP.Tree(rng=self.rng, primitive_set=self.primitive_set, terminal_set=self.terminal_set,
                        tree=eq, actual_lisp=True)
            f = eval('lambda x:' + t.convert_lisp_to_standard_for_function_creation())

            f([1])  # try to evaluate it at x=1. 
            self.f_hat = f
            return True

        except SyntaxError:
            return False
        except ValueError:
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

        return lisp


    def set_weights(self, weights):
        """Set the weights of a neural network built with
        keras. Excludes bias weights.

        Parameters
        ----------
        weights : list
            Flat version of the weights, which must be of the
            correct size.
        """

        weight_shapes = [(0,0)]
        start = 0

        for layer in self.network.layers:

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


    def get_weights(self):
        """Get the weights of a neural network built with
        keras. Excludes bias weights.
        """

        weights = []

        for layer in self.network.layers:

            layer_weights = layer.get_weights()

            if len(layer_weights) == 2:

                weights.extend(layer_weights[0].flatten())

            elif len(layer_weights) == 3:

                weights.extend(layer_weights[0].flatten())
                weights.extend(layer_weights[1].flatten())

        return weights


    def get_num_weights(self):
        """Count the number of weights
        necessary for set_weights. Does not
        count bias weights.
        """

        # get number of weights
        num_weights = 0
        for layer in self.network.layers:
            layer_weights = layer.get_weights()

            if len(layer_weights) > 0:
                for lw in layer_weights[:-1]:
                    num_weights += np.prod(lw.shape)

        return num_weights


    @staticmethod
    def get_lisp_summary(lisp, primitive_set, terminal_set):

        counts = {key: 0 for key in primitive_set + terminal_set}
        counts['unique subtrees under -'] = 0

        for char in lisp.split(' '):

            char = char.replace(')', '').replace('(', '')

            if '#f' in terminal_set and char not in primitive_set + terminal_set:
                counts['#f'] += 1
            else:
                counts[char] += 1

        
        if counts['-'] > 0:
            t = GP.Tree(rng=None, primitive_set=primitive_set, terminal_set=terminal_set,
                        tree=lisp, actual_lisp=True)
            
            node_map = t.get_node_map()
            
            # node_map['-'] = set of locations with - label
            for loc in node_map['-']:
                
                loc_left = (*loc, 0)
                loc_right = (*loc, 1)
                
                lisp_left = t.get_lisp_string(subtree=t.select_subtree(loc_left))
                lisp_right = t.get_lisp_string(subtree=t.select_subtree(loc_right))
                
                if lisp_left != lisp_right:
                    counts['unique subtrees under -'] += 1

        return counts


if __name__ == '__main__':

    options = {'use_k-expressions': True,
               'head_length': 3}    

    model = TlcsrNetwork(rng=np.random.RandomState(0),
                         num_data_encoder_inputs=2,
                         primitive_set=['*', '+', '-'],
                         terminal_set=['x0'],
                         timelimit=10,
                         options=options)

    x = np.linspace(-1, 1, 20)[:,None]
    f = lambda x: x[0]**2
    y = f(x.T)
    f_hat = lambda x: 0*x[0]
    f_hat_seq = ['START', '-', 'x0', 'x0']


    output = model.evaluate(x, y, f_hat, f_hat_seq,
                            return_decoded_list=True)

    print('best error', output['error_best'])
    print('final nn output', output['raw_decoded_list'])

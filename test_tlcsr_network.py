from TlcsrNetwork import TlcsrNetwork as Network

import numpy as np

def test_get_data_encoder_input_data():

    x = np.array([-1, 0, 1])[:,None]
    y = np.array([2, 4, 3])
    f_hat = lambda x: x[0]

    de_input = Network.get_data_encoder_input_data(x, y, f_hat)

    ans = [[[-1, 3],
            [0, 4],
            [1, 2]]]

    assert np.all(ans == de_input), 'Failed test_get_data_encoder_input_data'


def test_get_data_encoder_input_data_2():

    x = np.array([[-1, -2], [0, 0], [1,2]])
    y = np.array([2, 4, 3])
    f_hat = lambda x: x[0]+x[1]

    de_input = Network.get_data_encoder_input_data(x, y, f_hat)

    ans = [[[-1, -2, 5],
            [0, 0, 4],
            [1, 2, 0]]]

    assert np.all(ans == de_input), 'Failed test_get_data_encoder_input_data_2'


def test_get_lisp_from_stripped_lisp():

    primitives = ['*', '+', '-']

    stripped_lisps = ['x0', '* x0 x0', '+ - x0 * x0 x0 x0']
    answers = ['x0', '(* x0 x0)', '(+ (- x0 (* x0 x0)) x0)']

    for i, (stripped_lisp, ans) in enumerate(zip(stripped_lisps, answers)):
        lisp = Network.get_lisp_from_stripped_lisp(stripped_lisp, primitives)
        yield check_lisp, lisp, ans


def check_lisp(lisp, ans):
    assert lisp == ans, 'Failed on test '+lisp


class test_no_constants():

    def setup(self):

        self.N = Network(rng=np.random.RandomState(0),
                         num_data_encoder_inputs=2,
                         primitive_set=['*', '+', '-', '%'],
                         terminal_set=['x0'],
                         timelimit=100, use_constants=False,
                         options={'use_k-expressions': True,
                                  'head_length': 15})


    def test_get_eq_encoder_input_data(self):
        f_hat_seq = ['+', 'x0', 'x0']
        eqe_input = self.N.get_eq_encoder_input_data(f_hat_seq)

        #[START, STOP, primitive_set_elements, terminal_set_elements]
        ans =[[[0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 1],
               [0, 0, 0, 0, 0, 0, 1]]]

        assert np.all(ans == eqe_input)


    def test_read_decoded_output(self):
        outputs = [[[0, 0, 0, 1, 0, 0, 0], # +
                    [0, 0, 0, 0, 1, 0, 0], # -
                    [0, 0, 0, 0, 1, 0, 0], # -
                    [0, 0, 0, 0, 0, 0, 1], # x0
                    [0, 0, 0, 0, 0, 1, 0], # %
                    [0, 0, 0, 0, 0, 1, 0], # %
                    [0, 0, 0, 0, 0, 1, 0], # %
                    [0, 0, 0, 0, 0, 0, 1], # x0
                    [0, 0, 0, 0, 0, 0, 1], # x0
                    [0, 0, 0, 0, 0, 0, 1], # x0
                    [0, 0, 1, 0, 0, 0, 0], # *
                    [0, 0, 0, 0, 0, 0, 1], # x0
                    [0, 0, 1, 0, 0, 0, 0], # *
                    [0, 0, 0, 0, 0, 0, 1], # x0
                    [0, 0, 0, 0, 0, 0, 1], # x0
                    [0, 0, 1, 0, 0, 0, 0], # *
                    [0, 0, 0, 0, 0, 0, 1], # x0
                    [0, 0, 0, 0, 0, 0, 1], # x0
                    [0, 0, 0, 0, 0, 0, 1]  # x0
                    ]]
        decoded_output = self.N.read_decoded_output(outputs)

        ans = 'START + - - x0 % % % x0 x0 x0 * x0 * x0 x0 * x0 x0 x0'

        assert np.all(decoded_output == ans), 'Failed test_read_decoded_output'


    def test_get_decoder_input_data(self):

        decoder_input = self.N.get_decoder_input_data()

        ans = [[[1, 0, 0, 0, 0, 0, 0]]]
        assert np.all(ans == decoder_input)


    def test_evaluate(self):
        
        num_weights = self.N.get_num_weights()
        np.random.seed(0)
        weights = np.random.uniform(-1, 1, size=num_weights)
        self.N.set_weights(weights)

        x = np.linspace(-1, 1, 20)[:, None]
        target = lambda x: x[0]
        y = target(x.T)
        f_hat_seq = ['-', 'x0', 'x0']
        f_hat = lambda x: 0*x[0]

        output = self.N.evaluate(x, y, f_hat, f_hat_seq,
                                 return_equation=False,
                                 return_equation_str=True,
                                 return_decoded_list=True,
                                 return_errors=True)

        ans = {'error': 0.6865271624168707,
               'equation_str': ['x0', '(* (* (* (* x0 x0) (* x0 x0)) (* (* x0 x0) (* x0 x0))) (* (* (* x0 x0) (* x0 x0)) (* (* x0 x0) (* x0 x0))))', 'x0', '(* (* (* (* x0 x0) (* x0 x0)) (* (* x0 x0) (* x0 x0))) (* (* (* x0 x0) (* x0 x0)) (* (* x0 x0) (* x0 x0))))', 'x0', '(* (* (* (* x0 x0) (* x0 x0)) (* (* x0 x0) (* x0 x0))) (* (* (* x0 x0) (* x0 x0)) (* (* x0 x0) (* x0 x0))))', 'x0', '(* (* (* (* x0 x0) (* x0 x0)) (* (* x0 x0) (* x0 x0))) (* (* (* x0 x0) (* x0 x0)) (* (* x0 x0) (* x0 x0))))', 'x0', '(* (* (* (* x0 x0) (* x0 x0)) (* (* x0 x0) (* x0 x0))) (* (* (* x0 x0) (* x0 x0)) (* (* x0 x0) (* x0 x0))))'],
               'decoded_list': ['*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0'],
               'raw_decoded_list': ['*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0'],
               'errors': [0.0, 0.6865271624168707, 0.0, 0.6865271624168707, 0.0, 0.6865271624168707, 0.0, 0.6865271624168707, 0.0, 0.6865271624168707],
               'error_sum': 3.432635812084354,
               'error_best': 0.0,
               'equation_best': 'x0'}

        assert output == ans


    def test_rewrite_equation(self):

        num_weights = self.N.get_num_weights()
        np.random.seed(0)
        weights = np.random.uniform(-1, 1, size=num_weights)
        self.N.set_weights(weights)

        x = np.linspace(-1, 1, 20)[:, None]
        target = lambda x: x[0]
        y = target(x.T)
        f_hat_seq = ['-', 'x0', 'x0']
        f_hat = lambda x: 0*x[0]

        output = self.N.rewrite_equation(x, y, f_hat, f_hat_seq,
                                         initial_states=np.zeros((1,8)),
                                         return_equation=False,
                                         return_equation_str=True,
                                         return_decoded_list=True)

        ans = {'error': 0.0, 
               'equation_str': 'x0',
               'decoded_list': ['x0', '*', 'x0', 'x0', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0'], 'raw_decoded_list': ['x0', '*', 'x0', 'x0', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0']}

        assert output == ans


    def test_is_equation(self):

        lisps = ['x0', '(* x0 x0)', '(+ (- x0 (* x0 x0)) x0)',
                 '3', '(* x0)', '*', '(x0 x0 *)', '(sin x0)']
        answers = [True, True, True,
                   False, False, False, False, False]
        
        for i, (lisp, ans) in enumerate(zip(lisps, answers)):
            yield self.check_is_equation, lisp, ans


    def check_is_equation(self, lisp, ans):
        assert self.N.is_equation(lisp) == ans, 'Failure on test '+lisp


    def test_get_network_output_and_set_weights(self):
        num_weights = self.N.get_num_weights()
        weights = np.zeros(num_weights)
        self.N.set_weights(weights)

        output1 = self.N.get_network_output(initial_states=np.zeros((1,8)),
                                         data_encoder_input_data=np.zeros((1,5,2)),
                                         eq_encoder_input_data=np.zeros((1, 5, 2+len(self.N.primitive_set)+len(self.N.terminal_set))),
                                         decoder_input_data=self.N.get_decoder_input_data())
        
        x = np.linspace(-1, 1, 20)[:, None]
        target = lambda x: x[0]
        y = target(x.T)
        f_hat_seq = ['-', 'x0', 'x0']
        f_hat = lambda x: 0*x[0]

        output2 = self.N.get_network_output(initial_states=np.zeros((1,8)),
                                          data_encoder_input_data=self.N.get_data_encoder_input_data(x, y, f_hat),
                                          eq_encoder_input_data=self.N.get_eq_encoder_input_data(f_hat_seq),
                                          decoder_input_data=self.N.get_decoder_input_data())

        for i, output in enumerate([output1, output2]):
            assert len(np.unique(output)) == 1, 'Failure on test '+str(i)

    def test_set_get_weights(self):
        num_weights = self.N.get_num_weights()
        set_weights = np.random.uniform(-1, 1, num_weights)
        set_weights = set_weights.astype(np.float32)

        self.N.set_weights(set_weights)
        get_weights = self.N.get_weights()

        assert np.all(set_weights == get_weights)


class test_with_constants():

    def setup(self):

        self.N = Network(rng=np.random.RandomState(0),
                         num_data_encoder_inputs=2,
                         primitive_set=['*', '+', '-', '%'],
                         terminal_set=['x0', '#f'],
                         timelimit=100, use_constants=True,
                         options={'use_k-expressions': True,
                                  'head_length': 15})


    def test_get_eq_encoder_input_data(self):
        f_hat_seq = ['+', '-1', '0.5']
        eqe_input = self.N.get_eq_encoder_input_data(f_hat_seq)

        #[START, STOP, primitive_set_elements, terminal_set_elements]
        ans =[[[0, 0, 0, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 1, -1],
               [0, 0, 0, 0, 0, 0, 0, 1, 0.5]]]

        assert np.all(ans == eqe_input)


    def test_read_decoded_output(self):
        outputs = [[[0, 0, 1, 0, 0, 0, 0, 0], # *
                    [0, 0, 0, 0, 0, 0, 0, 1], # 2.718
                    [0, 0, 0, 1, 0, 0, 0, 0], # +
                    [0, 0, 0, 0, 1, 0, 0, 0], # -
                    [0, 0, 0, 0, 1, 0, 0, 0], # -
                    [0, 0, 0, 0, 0, 0, 1, 0], # x0
                    [0, 0, 0, 0, 0, 1, 0, 0], # %
                    [0, 0, 0, 0, 0, 1, 0, 0], # %
                    [0, 0, 0, 0, 0, 1, 0, 0], # %
                    [0, 0, 0, 0, 0, 0, 1, 0], # x0
                    [0, 0, 0, 0, 0, 0, 1, 0], # x0
                    [0, 0, 0, 0, 0, 0, 1, 0], # x0
                    [0, 0, 1, 0, 0, 0, 0, 0], # *
                    [0, 0, 0, 0, 0, 0, 1, 0], # x0
                    [0, 0, 1, 0, 0, 0, 0, 0], # *
                    [0, 0, 0, 0, 0, 0, 1, 0], # x0
                    [0, 0, 0, 0, 0, 0, 1, 0], # x0
                    [0, 0, 1, 0, 0, 0, 0, 0], # *
                    [0, 0, 0, 0, 0, 0, 1, 0], # x0
                    [0, 0, 0, 0, 0, 0, 1, 0], # x0
                    [0, 0, 0, 0, 0, 0, 0, 1]  # x0
                    ]]

        const_outputs = [[['0'],
                          ['2.718'],
                          ['0'],
                          ['0'],
                          ['0'],
                          ['0'],
                          ['0'],
                          ['0'],
                          ['0'],
                          ['0'],
                          ['0'],
                          ['0'],
                          ['0'],
                          ['0'],
                          ['0'],
                          ['0'],
                          ['0'],
                          ['0'],
                          ['0'],
                          ['0'],
                          ['3.1415926']]]

        decoded_output = self.N.read_decoded_output(outputs, const_outputs)

        ans = 'START * 2.718 + - - x0 % % % x0 x0 x0 * x0 * x0 x0 * x0 x0 3.1415926'

        assert np.all(decoded_output == ans)


    def test_get_decoder_input_data(self):

        decoder_input = self.N.get_decoder_input_data()

        ans = [[[1, 0, 0, 0, 0, 0, 0, 0, 0]]]
        assert np.all(ans == decoder_input)


    def test_is_equation(self):

        lisps = ['x0', '1.0', '(* x0 x0)', '(+ (- x0 (* x0 x0)) x0)',
                 '(* x0)', '*', '(x0 x0 *)', '(sin x0)']
        answers = [True, True, True, True,
                   False, False, False, False]
        
        for i, (lisp, ans) in enumerate(zip(lisps, answers)):
            yield self.check_is_equation, lisp, ans


    def check_is_equation(self, lisp, ans):
        assert self.N.is_equation(lisp) == ans, 'Failure on test '+lisp


    def test_get_network_output_and_set_weights(self):
        num_weights = self.N.get_num_weights()
        weights = np.zeros(num_weights)
        self.N.set_weights(weights)

        output1, const_outputs1 = self.N.get_network_output(initial_states=np.zeros((1,8)),
                                                           data_encoder_input_data=np.zeros((1,5,2)),
                                                           eq_encoder_input_data=np.zeros((1, 5, 2+len(self.N.primitive_set)+len(self.N.terminal_set)+1)),
                                                           decoder_input_data=self.N.get_decoder_input_data())
        
        x = np.linspace(-1, 1, 20)[:, None]
        target = lambda x: x[0]
        y = target(x.T)
        f_hat_seq = ['-', 'x0', 'x0']
        f_hat = lambda x: 0*x[0]

        data_encoder_input_data=self.N.get_data_encoder_input_data(x, y, f_hat)
        eq_encoder_input_data=self.N.get_eq_encoder_input_data(f_hat_seq)
        decoder_input_data=self.N.get_decoder_input_data()
        print(data_encoder_input_data.shape)
        print(eq_encoder_input_data.shape)
        print(decoder_input_data.shape)

        output2, const_output2 = self.N.get_network_output(initial_states=np.zeros((1,8)),
                                                           data_encoder_input_data=self.N.get_data_encoder_input_data(x, y, f_hat),
                                                           eq_encoder_input_data=self.N.get_eq_encoder_input_data(f_hat_seq),
                                                           decoder_input_data=self.N.get_decoder_input_data())

        for i, output in enumerate([output1, output2]):
            assert len(np.unique(output)) == 1, 'Failure on test '+str(i)


    def test_set_get_weights(self):
        num_weights = self.N.get_num_weights()
        set_weights = np.random.uniform(-1, 1, num_weights)
        set_weights = set_weights.astype(np.float32)

        self.N.set_weights(set_weights)
        get_weights = self.N.get_weights()

        assert np.all(set_weights == get_weights)


    def test_evaluate(self):
        
        num_weights = self.N.get_num_weights()
        np.random.seed(0)
        weights = np.random.uniform(-1, 1, size=num_weights)
        self.N.set_weights(weights)

        x = np.linspace(-1, 1, 20)[:, None]
        target = lambda x: x[0]
        y = target(x.T)
        f_hat_seq = ['-', 'x0', 'x0']
        f_hat = lambda x: 0*x[0]

        output = self.N.evaluate(x, y, f_hat, f_hat_seq,
                                 return_equation=False,
                                 return_equation_str=True,
                                 return_decoded_list=True,
                                 return_errors=True)

        ans = {'error': 2.0900768054383976,
               'equation_str': ['(+ (+ (% (% x0 x0) (% x0 x0)) (+ (% x0 x0) (% x0 x0))) (+ (% (% x0 x0) (* x0 x0)) (+ (% x0 x0) (% x0 x0))))', '(+ (% (% (% x0 x0) (% x0 x0)) (% (% x0 x0) (% x0 x0))) (% (% (% x0 x0) (% x0 x0)) (% (% x0 x0) (% x0 x0))))', '(+ (% (% (% x0 x0) (% x0 x0)) (% (% x0 x0) (% x0 x0))) (% (% (% x0 x0) (% x0 x0)) (% (% x0 x0) (% x0 x0))))', '(+ (% (% (% x0 x0) (% x0 x0)) (% (% x0 x0) (% x0 x0))) (% (% (% x0 x0) (% x0 x0)) (% (% x0 x0) (% x0 x0))))', '(+ (% (% (% x0 x0) (% x0 x0)) (% (% x0 x0) (% x0 x0))) (% (% (% x0 x0) (% x0 x0)) (% (% x0 x0) (% x0 x0))))', '(+ (% (% (% x0 x0) (% x0 x0)) (% (% x0 x0) (% x0 x0))) (% (% (% x0 x0) (% x0 x0)) (% (% x0 x0) (% x0 x0))))', '(+ (% (% (% x0 x0) (% x0 x0)) (% (% x0 x0) (% x0 x0))) (% (% (% x0 x0) (% x0 x0)) (% (% x0 x0) (% x0 x0))))', '(+ (% (% (% x0 x0) (% x0 x0)) (% (% x0 x0) (% x0 x0))) (% (% (% x0 x0) (% x0 x0)) (% (% x0 x0) (% x0 x0))))', '(+ (% (% (% x0 x0) (% x0 x0)) (% (% x0 x0) (% x0 x0))) (% (% (% x0 x0) (% x0 x0)) (% (% x0 x0) (% x0 x0))))', '(+ (% (% (% x0 x0) (% x0 x0)) (% (% x0 x0) (% x0 x0))) (% (% (% x0 x0) (% x0 x0)) (% (% x0 x0) (% x0 x0))))'],
               'decoded_list': ['+', '%', '%', '%', '%', '%', '%', '%', '%', '%', '%', '%', '%', '%', '%', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0'],
               'raw_decoded_list': ['+', '%', '%', '%', '%', '%', '%', '%', '%', '%', '%', '%', '%', '%', '%', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0'],
               'errors': [116.98218767725601, 2.0900768054383976, 2.0900768054383976, 2.0900768054383976, 2.0900768054383976, 2.0900768054383976, 2.0900768054383976, 2.0900768054383976, 2.0900768054383976, 2.0900768054383976],
               'error_sum': 135.7928789262016,
               'error_best': 2.0900768054383976,
               'equation_best': '(+ (% (% (% x0 x0) (% x0 x0)) (% (% x0 x0) (% x0 x0))) (% (% (% x0 x0) (% x0 x0)) (% (% x0 x0) (% x0 x0))))'}

        assert output == ans


    def test_rewrite_equation(self):

        num_weights = self.N.get_num_weights()
        np.random.seed(0)
        weights = np.random.uniform(-1, 1, size=num_weights)
        self.N.set_weights(weights)

        x = np.linspace(-1, 1, 20)[:, None]
        target = lambda x: x[0]
        y = target(x.T)
        f_hat_seq = ['-', 'x0', 'x0']
        f_hat = lambda x: 0*x[0]

        output = self.N.rewrite_equation(x, y, f_hat, f_hat_seq,
                                         initial_states=np.zeros((1,8)),
                                         return_equation=False,
                                         return_equation_str=True,
                                         return_decoded_list=True)

        ans = {'error': 116.98218767725601,
               'equation_str': '(+ (+ (% (% x0 x0) (% x0 x0)) (+ (% x0 x0) (% x0 x0))) (+ (% (% x0 x0) (* x0 x0)) (+ (% x0 x0) (% x0 x0))))',
               'decoded_list': ['+', '+', '+', '%', '+', '%', '+', '%', '%', '%', '%', '%', '*', '%', '%', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0'],
               'raw_decoded_list': ['+', '+', '+', '%', '+', '%', '+', '%', '%', '%', '%', '%', '*', '%', '%', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0', 'x0']}
        assert output == ans


def test_get_lisp_summary():

    primitive_set = ['*', '+', '-']
    terminal_set = ['x0', '#f']

    lisps = ['x0',
             '1',
             '(* x0 x0)',
             '(- x0 x0)',
             '(- x0 (+ x0 x0))']

    answers = [{'*': 0, '+': 0, '-': 0, 'x0': 1, '#f': 0, 'unique subtrees under -': 0},
               {'*': 0, '+': 0, '-': 0, 'x0': 0, '#f': 1, 'unique subtrees under -': 0},
               {'*': 1, '+': 0, '-': 0, 'x0': 2, '#f': 0, 'unique subtrees under -': 0},
               {'*': 0, '+': 0, '-': 1, 'x0': 2, '#f': 0, 'unique subtrees under -': 0},
               {'*': 0, '+': 1, '-': 1, 'x0': 3, '#f': 0, 'unique subtrees under -': 1}]

    for lisp, ans in zip(lisps, answers):
        output = Network.get_lisp_summary(lisp=lisp,
                                          primitive_set=primitive_set,
                                          terminal_set=terminal_set)
        yield check_output, output, ans, 'Failure on '+lisp

def check_output(output, ans, msg='Failure'):
    assert output == ans, msg

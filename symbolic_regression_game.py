import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gym

import itertools

import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K


class SymbolicRegressionGame(gym.Env):

    def __init__(self, rng, x, y, f, time_limit, tree=None, f_val=None, x_val=None, f_test=None, x_test=None, y_test=None,
                 target=None, model=None, extra_constant=False):

        super(SymbolicRegressionGame, self).__init__()
        
        self.rng = rng
        self.model = model
        self.extra_constant = extra_constant

        # initial shift amount
        self.v_original = 0

        self.x = x
        self.y_original = y
        self.f = f

        if x_val is not None:
            self.x_val = x_val
            self.f_val = f_val
        
        if x_test is not None:
            self.x_test = x_test
            self.f_test = f_test
            self.y_test = y_test

        self.time_limit = time_limit
        
        self.FLoPs = 0
        
        self.tree = tree
        self.target = target

        self.reset()

        # This is the min and max of output space.
        self.action_space = gym.spaces.Box(np.array([-1]), np.array([+1]))  # vshift

        # State space
        lower_bounds = np.array([-1, -1, -np.inf]*len(x[0]))
        upper_bounds = -1*lower_bounds
        self.observation_space = gym.spaces.Box(lower_bounds, upper_bounds)  # xi, yi, f(xi), ...

        self.num_envs = 1

        self.num_hidden = int(np.mean([len(lower_bounds), self.action_space.shape[0]]))
        self.num_layers = 1

        self.errors = []
        self.initial_shifts = []
        self.final_shifts = []
        self.val_errors = []
        self.test_errors = []

    
    def reset(self, v=None, s=None, datatype='train'):

        # hidden_states = K.variable(value=np.zeros((3, 1)))
        # cell_states = K.variable(value=np.zeros((3, 1)))

        # self.model.layers[0].states[0] = hidden_states

        self.FLoPs += len(self.x)*self.get_FLoPs_tree()

        if v is None:
            self.v_original = self.rng.uniform(-5, 5)

        if s is None:
            self.s_original = self.rng.uniform(-5, 5)
    
        self.s = 1
        self.v = 0

        if self.extra_constant:
            self.y = self.f(self.x, self.v_original, self.s_original)

            if hasattr(self, 'f_val'):
                self.y_val = self.f_val(self.x_val, self.v_original, self.s_original)
        
            # if hasattr(self, 'f_test'):
            #     self.y_test = self.f_test(self.x_test, self.v_original, self.s_original)

        else:
            self.y = self.f(self.x, self.v_original)

            if hasattr(self, 'f_val'):
                self.y_val = self.f_val(self.x_val, self.v_original)
        
            # if hasattr(self, 'f_test'):
            #     self.y_test = self.f_test(self.x_test, self.v_original)

        self.timestep = 0

        return self.get_state(datatype)


    def _step(self, action, datatype='train'):

        if self.extra_constant:
            v_change, s_change = action[0]
            self.update_s(s_change)

        else:
            v_change = action

        while type(v_change) not in (np.float32, np.float64, float):
            v_change = v_change[0]

        self.update_v(v_change)

        r = self.get_reward(datatype)

        new_obs = self.get_state(datatype)

        done = [True] if self.timestep > self.time_limit else [False]
        done = np.array(done)

        info = {}

        # if done == [True] and datatype == 'train':
        #     self.errors.append(self.get_error(datatype))
        #     self.initial_shifts.append(self.v_original)
        #     self.final_shifts.append(self.v)

        #     self.reset()

        return new_obs, r, done, info


    def step(self, action):

        return self._step(action, datatype='train')


    def update_v(self, change_in_v):

        self.FLoPs += 1
        self.v += change_in_v
        self.timestep += 1


    def update_s(self, change_in_s):

        self.FLoPs += 1
        self.s += change_in_s


    # def get_signed_error(self):

    #     return np.mean(self.y - self.f(self.x, self.v))


    def get_error(self, datatype):

        self.FLoPs += (3+self.get_FLoPs_tree())*len(self.x)

        if datatype == 'train':
            if self.extra_constant:
                return np.sqrt(np.mean(np.power(self.y - self.f(self.x, self.v, self.s), 2)))

            else:
                return np.sqrt(np.mean(np.power(self.y - self.f(self.x, self.v), 2)))

        elif datatype == 'test':
            if self.extra_constant:
                return np.sqrt(np.mean(np.power(self.y_test - self.f_test(self.x_test, self.v, self.s), 2)))

            else:
                return np.sqrt(np.mean(np.power(self.y_test - self.f_test(self.x_test, self.v), 2)))

    def get_reward(self, datatype):

        self.FLoPs += 1

        return np.array([-self.get_error(datatype)])


    def get_inputs(self, datatype):

        self.FLoPs += len(self.x)*(1+self.get_FLoPs_tree())

        if datatype == 'train':
            # return np.array(list(itertools.chain(*[[*xi, yi - self.f(xi, self.v)] for xi, yi in zip(self.x.T, self.y)])))
            if self.extra_constant:
                return np.array([[*xi, yi - self.f(xi, self.v, self.s)] for xi, yi in zip(self.x.T, self.y)])

            else:
                return np.array([[*xi, yi - self.f(xi, self.v)] for xi, yi in zip(self.x.T, self.y)])


        elif datatype == 'validation':
            # return np.array(list(itertools.chain(*[[*xi, yi - self.f_val(xi, self.v)] for xi, yi in zip(self.x_val.T, self.y_val)])))
            if self.extra_constant:
                return np.array([[*xi, yi - self.f_val(xi, self.v, self.s)] for xi, yi in zip(self.x_val.T, self.y_val)])

            else:
                return np.array([[*xi, yi - self.f_val(xi, self.v)] for xi, yi in zip(self.x_val.T, self.y_val)])

        elif datatype == 'test':
            # return np.array(list(itertools.chain(*[[*xi, yi - self.f_test(xi, self.v)] for xi, yi in zip(self.x_test.T, self.y_test)])))
            if self.extra_constant:
                return np.array([[*xi, yi - self.f_test(xi, self.v, self.s)] for xi, yi in zip(self.x_test.T, self.y_test)])

            else:
                return np.array([[*xi, yi - self.f_test(xi, self.v)] for xi, yi in zip(self.x_test.T, self.y_test)])


    def get_state(self, datatype):

        return self.get_inputs(datatype)[None, :]


    def close(self):
        pass


    def get_next_action(self, datatype):

        state = self.get_state(datatype)
        next_action = self.model.predict(state)
        # print(state.shape)
        # exit()

        return next_action


    def render(self, mode=None):

        if self.timestep == 0:

            plt.close('all')

            plt.ion()
            self.fig = plt.figure()

            if self.x.shape[0] > 1:
                self.curve, = plt.plot(self.x[0], self.f(self.x, self.v), 'C0', label='$\\hat{f}(x)$')
                plt.plot(self.x[0], self.y, 'C1', label='$y$')
            
            else:
                self.curve, = plt.plot(self.x, self.f(self.x, self.v), 'C0', label='$\\hat{f}(x)$')
                plt.plot(self.x, self.y, 'C1', label='$y$')
            
            plt.legend()
            plt.xlabel('$x$')
            plt.ylabel('$y$')
            plt.ylim([-5, 5])
            plt.title('t = '+str(self.timestep))

        self.curve.set_ydata(self.f(self.x, self.v))
        plt.title('target = '+str(self.target)+', t = '+str(self.timestep)+', RMSE = {:.3g}'.format(self.get_error('train')) )
        self.fig.canvas.draw()
        plt.pause(0.5)


    def save(self, path):

        df = pd.DataFrame(np.array([self.errors, self.initial_shifts, self.final_shifts]).T)
        df.to_csv(path, index=False, header=['Errors', 'Initial Shift', 'Final Shift'])


    def get_FLoPs_tree(self):

        num_leaves, num_nodes = self.tree.get_num_leaves(num_nodes=True)
        num_nonleaves = num_nodes - num_leaves

        return num_nonleaves


    def clear_FLoPs(self):

        self.FLoPs = 0


    def set_weights(self, weights):

        weight_shapes = [(0,0)]

        for layer in self.model.layers:

            layer_weights = layer.get_weights()

            if len(layer_weights) == 3:

                weight_shapes.extend([layer_weights[0].shape, layer_weights[1].shape])

                start = np.prod(weight_shapes[-3])
                end1 = start + np.prod(weight_shapes[-2])
                end2 = end1 + np.prod(weight_shapes[-1])

                new_weights = np.array([weights[start:end1].reshape(weight_shapes[-2]),
                                        weights[end1:end2].reshape(weight_shapes[-1]),
                                        np.zeros(weight_shapes[-1][0])])
                layer.set_weights(new_weights)


    def play_game(self, weights, v, s, datatype='train', final_error_only=False):

        if weights is not None:
            self.set_weights(weights)

        state = self.reset(v, s, datatype)

        state = self.get_state(datatype)

        neg_reward_sum = 0
        done = False

        while not done:

            action = self.get_next_action(datatype)
            state, reward, done, _ = self._step(action, datatype)
            neg_reward_sum += -reward

        if final_error_only:
            return -reward

        else:
            return neg_reward_sum


if __name__ == '__main__':

    import GeneticProgramming as GP

    import numpy as np
    from keras.optimizers import Adam
    from keras.models import Sequential
    from keras.layers.core import Dense
    from keras.layers import SimpleRNN

    target_names = ['quartic', 'septic', 'nonic', 'keijzer11', 'keijzer12', 'keijzer13', 'keijzer14',
                    'keijzer15', 'r1', 'r2', 'r3']

    lisps = {'quartic': '(* (x0) (+ (1) (* (x0) (+ (1) (* (x0) (+ (1) (x0)))))))',
             'septic': '(* (x0) (+ (1) (* (x0) (+ (-2) (* (x0) (+ (1) (* (x0) (+ (-1) (* (x0) (+ (1) (* (x0) (+ (-2) (x0)))))))))))))',
             'nonic': '(* (x0) (+ (1) (* (x0) (+ (1) (* (x0) (+ (1) (* (x0) (+ (1) (* (x0) (+ (1) (* (x0) (+ (1) (* (x0) (+ (1) (* (x0) (+ (1) (x0)))))))))))))))))',
             'r1': '(% (* (+ (x0) (1)) (* (+ (x0) (1)) (+ (x0) (1)))) (+ (1) (* (x0) (+ (-1) (x0)))))',
             'r2': '(% (+ (* (* (* (x0) (x0)) (x0)) (- (* (x0) (x0)) (3))) (1)) (+ (* (x0) (x0)) (1)))',
             'r3': '(% (* (* (* (* (x0) (x0)) (x0)) (* (x0) (x0))) (+ (1) (x0))) (+ (1) (* (x0) (+ (1) (* (x0) (+ (1) (* (x0) (+ (1) (x0)))))))))',
             'keijzer11': '(+ (* (x0) (x1)) (sin (* (- (x0) (1)) (- (x1) (1)))))',
             'keijzer12': '(+ (* (* (* (x0) (x0)) (x0)) (- (x0) (1))) (* (x1) (- (* (0.5) (x1)) (1))))',
             'keijzer13': '(* (6) (* (sin (x0)) (cos (x1))))',
             'keijzer14': '(% (8) (+ (2) (+ (* (x0) (x0)) (* (x1) (x1)))))',
             'keijzer15': '(+ (* (x0) (- (* (0.2) (* (x0) (x0))) (1))) (* (x1) (- (* (0.5) (* (x1) (x1))) (1))))'}

    benchmark_index = 0
    jumbled_target_name_indices = [(benchmark_index+i+1) % len(target_names)  for i, _ in enumerate(target_names)] 
    jumbled_target_name = [target_names[i] for i in jumbled_target_name_indices]
    print(target_names)
    print(jumbled_target_name)

    tree_test = GP.Tree(rng=None, tree='(+ '+lisps[jumbled_target_name[-1]]+' (0))', num_vars=2)
    f_test = eval('lambda x, v: '+tree_test.convert_lisp_to_standard_for_function_creation()+'+v')
    x_test = np.vstack((np.linspace(-1, 1, 20), np.linspace(-1, 1, 20)))

    # val not used
    f_val = lambda x, v: x**2 + v
    x_val = np.linspace(-1, 1, 20)[None, :]

    # get the tree, +0 for the v
    tree = GP.Tree(rng=None, tree='(+ '+lisps[jumbled_target_name[0]]+' (0))', num_vars=2)
    f = eval('lambda x, v: '+tree.convert_lisp_to_standard_for_function_creation()+'+v')
    x = np.vstack((np.linspace(-1, 1, 20), np.linspace(-1, 1, 20)))

    seed = 1
    rng = np.random.RandomState(seed)
    v = rng.uniform(-5, 5)
    y = f(x, v)

    time_limit = 10

    num_input = 3
    num_output = 1

    model = Sequential()
    model.add(SimpleRNN(num_output, batch_input_shape=(None, 20, num_input), return_sequences=False, activation='tanh'))

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

    SRG = SymbolicRegressionGame(rng=rng, x=x, y=y, f=f, time_limit=time_limit, tree=tree,
                                 f_val=f_val, x_val=x_val,
                                 f_test=f_test, x_test=x_test,
                                 target=jumbled_target_name[-1], model=model)

    sum_reward = SRG.play_game()
    print(sum_reward)

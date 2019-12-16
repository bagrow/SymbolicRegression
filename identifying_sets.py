import numpy as np

from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.layers.core import Dense
from keras.layers import Input, SimpleRNN

from sklearn.model_selection import train_test_split


np.random.seed(0)

width = num_data_points = 20
num_input = 2
overlap = 4

assert overlap < width, 'overlap must be less than width'

num_sets = 5

# generate data
initial_start = 0
starts = [initial_start]

for _ in range(num_sets-1):
    next_start = starts[-1] + width - overlap
    starts.append(next_start)

# sets of tuples of 2 elements
x_seq = list(range(width))
y_seq = list(range(int(width/2), int(width/2)+width))

base_set = [(x,y) for x, y in zip(x_seq, y_seq)]
switch_set = [(y,x) for x, y in zip(x_seq, y_seq)]

max_x = max(x_seq)
max_y = max(y_seq)
offset_both_set = [(x+max_x ,y+max_y) for x, y in zip(x_seq, y_seq)]
offset_one_set = [(x ,y+max_y) for x, y in zip(x_seq, y_seq)]

np.random.shuffle(y_seq)
shuffle_set = [(x,y) for x, y in zip(x_seq, y_seq)]

sets = np.array([base_set, shuffle_set, switch_set, offset_one_set, offset_both_set])

# sets = np.array([list({i for i in range(s, s+width)}) for s in starts])

x = [s for s in sets]
y = [np.eye(len(sets))[i] for i, _ in enumerate(sets)]

indices = np.array(list(range(len(sets[0]))))

num_copies_of_set = 1000

for _ in range(num_copies_of_set):
    np.random.shuffle(indices)
    for i, s in enumerate(sets):
        x.append(s[indices])
        y.append(np.eye(len(sets))[i])


# convert to recurrent input
x = np.array([[i for i in input_sequence] for input_sequence in x])

# split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# build model
num_output = len(sets)

model = Sequential()
model.add(SimpleRNN(8, input_shape=(num_data_points, num_input),
                    return_sequences=True, activation='relu'))
model.add(SimpleRNN(8,
                    return_sequences=False, activation='relu'))
model.add(Dense(num_output, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print([[x[0]]])
# model.fit([[x[0]]], [[y[0]]])   # works!
model.fit([x_train], [y_train], batch_size=len(x[0]), epochs=20, shuffle=True)

loss, accuracy = model.evaluate([x_test], [y_test])

print('test loss', loss, 'test  accuracy', accuracy)

# for o, yi in zip(output, y_test):

#     if np.argmax(o) != np.argmax(yi):
#         print('mistake!')

# print('If mistakes occured, the word will be printed above this line.')
import numpy as np

from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.layers.core import Dense
from keras.layers import Input, SimpleRNN


np.random.seed(0)

num_data_points = 5
num_input = 1

# generate data
sets = np.array([list({i for i in range(*n)}) for n in [(0, num_data_points), (num_data_points,2*num_data_points)]])

x = [s for s in sets]
y = [np.eye(len(sets))[i] for i, _ in enumerate(sets)]

indices = np.array(list(range(len(sets[0]))))

for _ in range(2000):
    np.random.shuffle(indices)
    for i, s in enumerate(sets):
        x.append(s[indices])
        y.append(np.eye(len(sets))[i])


# convert to recurrent input
x = np.array([[[i] for i in input_sequence] for input_sequence in x])

print(x.shape)
print(num_data_points, num_input)

# build model
num_output = len(sets)

model = Sequential()
model.add(SimpleRNN(5, input_shape=(num_data_points, num_input),
                    return_sequences=False, activation='relu'))
model.add(Dense(num_output, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print([[x[0]]])
# model.fit([[x[0]]], [[y[0]]])   # works!
model.fit([x], [y], batch_size=len(x[0]), epochs=20, shuffle=True)

output = model.predict([x])

for o, yi in zip(output, y):

    if np.argmax(o) != np.argmax(yi):
        print('mistake!')

print('If mistakes occured, the word will be printed above this line.')
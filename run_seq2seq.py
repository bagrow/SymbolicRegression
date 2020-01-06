from seq2seq_first import seq2seq

s2s = seq2seq(num_encoder_tokens=2,
              primitive_set=['*', '+'],
              terminal_set=['x0'],
              max_decoder_seq_length=30)

x = np.linspace(-1, 1, 20)[None, :]
f = lambda x: x[0]**2
y = f(x)
f_hat = lambda x: 0*x[0]

print('x.shape', x.shape)

decoded_string = s2s.evaluate(x, y, f_hat)
print('decoded_string', decoded_string)

cma_train(model, dataset, evaluate)


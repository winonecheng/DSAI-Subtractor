from keras.models import Sequential
from keras import layers
import numpy as np
import pandas as pd
from six.moves import range


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

class CharacterTable(object):
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
    
    def encode(self, C, num_rows):
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x
    
    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return "".join(self.indices_char[i] for i in x)

def run(args):
    TRAINING_SIZE = int(args.train)
    DIGITS = int(args.digits)
    ADD_SUB_MIX = True
    MAXLEN = DIGITS + 1 + DIGITS
    RNN = layers.LSTM
    HIDDEN_SIZE = 128
    BATCH_SIZE = 128
    LAYERS = int(args.layers)

    chars = '0123456789+- ' if ADD_SUB_MIX else '0123456789+'
    op_option = list('+-' if ADD_SUB_MIX else '+')
    ctable = CharacterTable(chars)

    questions = []
    expected = []
    seen = set()
    print('Generating data...')
    while len(questions) < TRAINING_SIZE:
        f = lambda: int(''.join(np.random.choice(list('0123456789')) for _ in range(DIGITS)))
        a, b, op = f(), f(), np.random.choice(op_option)
        key = a, b, op
        if key in seen or a < 10**(DIGITS-1) or b < 10**(DIGITS-1):
            continue
        seen.add(key)
        q = '{}{}{}'.format(a, op, b)
        query = q + ' ' * (MAXLEN - len(q))
        ans = str(a+b if op == '+' else a-b)
        ans += ' ' * (DIGITS + 1 - len(ans))
        questions.append(query)
        expected.append(ans)
    print('Total addition questions:', len(questions))

    # # Processing

    print('Vectorization...')
    x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
    y = np.zeros((len(expected), DIGITS + 1, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(questions):
        x[i] = ctable.encode(sentence, MAXLEN)
    for i, sentence in enumerate(expected):
        y[i] = ctable.encode(sentence, DIGITS + 1)


    indices = np.arange(len(y))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    # train_test_split
    split_at = len(x) - 10000
    train_x, test_x = x[:split_at], x[split_at:]
    train_y, test_y = y[:split_at], y[split_at:]

    split_at = len(train_x) - len(train_x) // 10
    (x_train, x_val) = train_x[:split_at], train_x[split_at:]
    (y_train, y_val) = train_y[:split_at], train_y[split_at:]

    print('Training Data:')
    print(x_train.shape)
    print(y_train.shape)

    print('Validation Data:')
    print(x_val.shape)
    print(y_val.shape)

    print('Testing Data:')
    print(test_x.shape)
    print(test_y.shape)


    print('Build model...')
    model = Sequential()
    model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
    model.add(layers.RepeatVector(DIGITS + 1))
    for _ in range(LAYERS):
        model.add(RNN(HIDDEN_SIZE, return_sequences=True))

    model.add(layers.TimeDistributed(layers.Dense(len(chars))))
    model.add(layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    model.summary()


    # # Training

    accuracy = list()
    history = list()

    for iteration in range(10):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        history.append(model.fit(x_train, y_train,
                                batch_size=BATCH_SIZE,
                                epochs=1,
                                validation_data=(x_val, y_val),
                                verbose=0))
        
        right = 0
        preds = model.predict_classes(test_x, verbose=0)
        for i in range(len(preds)):
            q = ctable.decode(test_x[i])
            correct = ctable.decode(test_y[i])
            guess = ctable.decode(preds[i], calc_argmax=False)
            if correct == guess:
                right += 1

        accuracy.append(right / len(preds))
        print("MSG : Accuracy is {}".format(right / len(preds)))


    # # Output

    print('Output...')
    df = pd.DataFrame([[hist.history['acc'][0], hist.history['val_acc'][0]] for hist in history], columns=['acc', 'val_acc'])
    df.insert(2, 'test_acc', accuracy)
    df.to_csv('output_{}_{}.csv'.format(x_train.shape[0], x_val.shape[0]), index=False)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train',
                       default='40000',
                       help='size of training data')
    parser.add_argument('--digits',
                        default='3',
                        help='digits')
    parser.add_argument('--layers',
                        default='1',
                        help='number of layers')
    args = parser.parse_args()

    run(args)

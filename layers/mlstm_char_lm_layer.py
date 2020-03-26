import html
import numpy as np
import tensorflow as tf


class Model(object):

    def __init__(self, weights_path, batch_size=128, max_seq_length=64, hidden_dim=4096):
        self.weights_path = weights_path
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.hidden_dim = hidden_dim
        self.nembd = 64
        self.nstates = 2
        self.nvocab = 256
        self.out_wn = False
        self.rnn_wn = True
        self.embd_wn = True
        self.n_loaded = 0

        # Load Weights
        self.weights = [np.load(self.weights_path + '/%d.npy' % i) for i in range(15)]
        self.weights[2] = np.concatenate(self.weights[2:6], axis=1)
        self.weights[3:6] = []

        self.X = tf.placeholder(tf.int32, [None, self.max_seq_length])
        self.M = tf.placeholder(tf.float32, [None, self.max_seq_length, 1])
        self.S = tf.placeholder(tf.float32, [self.nstates, None, self.hidden_dim])
        self.cells, self.states, self.logits = self.model(self.X, self.S, self.M, reuse=False)

        self.sess = tf.Session()
        tf.global_variables_initializer().run(session=self.sess)

    def seq_rep(self, xmb, mmb, smb):
        return self.sess.run(self.states, {self.X: xmb, self.M: mmb, self.S: smb})

    def seq_cells(self, xmb, mmb, smb):
        return self.sess.run(self.cells, {self.X: xmb, self.M: mmb, self.S: smb})

    def transform(self, xs):
        xs = [self.preprocess(x, self.max_seq_length) for x in xs]
        lens = np.asarray([len(x) for x in xs])
        sorted_idxs = np.argsort(lens)
        unsort_idxs = np.argsort(sorted_idxs)
        sorted_xs = [xs[i] for i in sorted_idxs]
        maxlen = np.max(lens)
        offset = 0
        n = len(xs)
        smb = np.zeros((2, n, self.hidden_dim), dtype=np.float32)
        for step in range(0, self.ceil_round_step(maxlen, self.max_seq_length), self.max_seq_length):
            start = step
            end = step + self.max_seq_length
            xsubseq = [x[start:end] for x in sorted_xs]
            ndone = sum([x == b'' for x in xsubseq])
            offset += ndone
            xsubseq = xsubseq[ndone:]
            sorted_xs = sorted_xs[ndone:]
            nsubseq = len(xsubseq)
            xmb, mmb = self.batch_pad(xsubseq, nsubseq, self.max_seq_length)
            for batch in range(0, nsubseq, self.batch_size):
                start = batch
                end = batch + self.batch_size
                batch_smb = self.seq_rep(xmb[start:end], mmb[start:end], smb[:, offset + start:offset + end, :])
                smb[:, offset + start:offset + end, :] = batch_smb
        features = smb[0, unsort_idxs, :]

        return features

    def cell_transform(self, xs, indexes=None):
        fs = []
        xs = [self.preprocess(x, self.max_seq_length) for x in xs]
        for xmb in self.iter_data(xs, size=self.batch_size):
            smb = np.zeros((2, self.batch_size, self.hidden_dim))
            n = len(xmb)
            xmb, mmb = self.batch_pad(xmb, self.batch_size, self.max_seq_length)
            smb = self.seq_cells(xmb, mmb, smb)
            smb = smb[:, :n, :]
            if indexes is not None:
                smb = smb[:, :, indexes]
            fs.append(smb)
        fs = np.concatenate(fs, axis=1).transpose(1, 0, 2)

        return fs

    def model(self, X, S, M=None, reuse=False):
        nsteps = X.get_shape()[1]
        cstart, hstart = tf.unstack(S, num=self.nstates)
        with tf.variable_scope('model', reuse=reuse):
            words = self.embd(X, self.nembd)
            inputs = tf.unstack(words, nsteps, 1)
            hs, cells, cfinal, hfinal = self.mlstm(inputs, cstart, hstart, M, self.hidden_dim, scope='rnn', wn=self.rnn_wn)
            hs = tf.reshape(tf.concat(hs, 1), [-1, self.hidden_dim])
            logits = self.fc(hs, self.nvocab, act=lambda x: x, wn=self.out_wn, scope='out')
        states = tf.stack([cfinal, hfinal], 0)
        return cells, states, logits

    def embd(self, X, ndim, scope='embedding'):
        with tf.variable_scope(scope):
            embd = tf.get_variable("w", [self.nvocab, ndim], initializer=self.load_params)
            h = tf.nn.embedding_lookup(embd, X)
            return h

    def mlstm(self, inputs, c, h, M, n_dim, scope='lstm', wn=False):
        n_in = inputs[0].get_shape()[1].value
        with tf.variable_scope(scope):
            wx = tf.get_variable("wx", [n_in, n_dim * 4], initializer=self.load_params)
            wh = tf.get_variable("wh", [n_dim, n_dim * 4], initializer=self.load_params)
            wmx = tf.get_variable("wmx", [n_in, n_dim], initializer=self.load_params)
            wmh = tf.get_variable("wmh", [n_dim, n_dim], initializer=self.load_params)
            b = tf.get_variable("b", [n_dim * 4], initializer=self.load_params)
            if wn:
                gx = tf.get_variable("gx", [n_dim * 4], initializer=self.load_params)
                gh = tf.get_variable("gh", [n_dim * 4], initializer=self.load_params)
                gmx = tf.get_variable("gmx", [n_dim], initializer=self.load_params)
                gmh = tf.get_variable("gmh", [n_dim], initializer=self.load_params)

        if wn:
            wx = tf.nn.l2_normalize(wx, dim=0) * gx
            wh = tf.nn.l2_normalize(wh, dim=0) * gh
            wmx = tf.nn.l2_normalize(wmx, dim=0) * gmx
            wmh = tf.nn.l2_normalize(wmh, dim=0) * gmh

        cs = []
        for idx, x in enumerate(inputs):
            m = tf.matmul(x, wmx) * tf.matmul(h, wmh)
            z = tf.matmul(x, wx) + tf.matmul(m, wh) + b
            i, f, o, u = tf.split(z, 4, 1)
            i = tf.nn.sigmoid(i)
            f = tf.nn.sigmoid(f)
            o = tf.nn.sigmoid(o)
            u = tf.tanh(u)
            if M is not None:
                ct = f * c + i * u
                ht = o * tf.tanh(ct)
                m = M[:, idx, :]
                c = ct * m + c * (1 - m)
                h = ht * m + h * (1 - m)
            else:
                c = f * c + i * u
                h = o * tf.tanh(c)
            inputs[idx] = h
            cs.append(c)
        cs = tf.stack(cs)
        return inputs, cs, c, h

    def fc(self, x, n_out, act, wn=False, bias=True, scope='fc'):
        with tf.variable_scope(scope):
            nin = x.get_shape()[-1].value
            w = tf.get_variable("w", [nin, n_out], initializer=self.load_params)
            if wn:
                g = tf.get_variable("g", [n_out], initializer=self.load_params)
            if wn:
                w = tf.nn.l2_normalize(w, dim=0) * g
            z = tf.matmul(x, w)
            if bias:
                b = tf.get_variable("b", [n_out], initializer=self.load_params)
                z = z + b
            h = act(z)
            return h

    def load_params(self, shape, dtype, *args, **kwargs):
        self.n_loaded += 1
        return self.weights[self.n_loaded - 1]

    @staticmethod
    def preprocess(text, max_seq_length, front_pad='\n ', end_pad=' '):
        text = text[0].decode('UTF-8')
        text = html.unescape(text)
        text = text.replace('\n', ' ').strip()
        # Truncate to max_seq_len - 3 to account for front/end padding
        text = text[:max_seq_length - 3] if len(text) > max_seq_length - 3 else text
        text = front_pad + text + end_pad
        text = text.encode()
        return text

    @staticmethod
    def iter_data(*data, **kwargs):
        size = kwargs.get('size', 128)
        try:
            n = len(data[0])
        except:
            n = data[0].shape[0]
        batches = n // size
        if n % size != 0:
            batches += 1

        for b in range(batches):
            start = b * size
            end = (b + 1) * size
            if end > n:
                end = n
            if len(data) == 1:
                yield data[0][start:end]
            else:
                yield tuple([d[start:end] for d in data])

    @staticmethod
    def ceil_round_step(n, step):
        return int(np.ceil(n / step) * step)

    @staticmethod
    def batch_pad(xs, n_batch, n_steps):
        xmb = np.zeros((n_batch, n_steps), dtype=np.int32)
        mmb = np.ones((n_batch, n_steps, 1), dtype=np.float32)
        for i, x in enumerate(xs):
            l = len(x)
            n_pad = n_steps - l
            xmb[i, -l:] = list(x)
            mmb[i, :n_pad] = 0
        return xmb, mmb


class MLSTMCharLMLayer(tf.keras.layers.Layer):
    """ Wraps the mLSTM Character Language Model in a Keras Layer.

    Radford, A., Jozefowicz, R. and Sutskever, I. (2018) ‘Learning to Generate Reviews and Discovering Sentiment’,
    arXiv. Available at: http://arxiv.org/abs/1704.01444
    https://github.com/openai/generating-reviews-discovering-sentiment
    """
    def __init__(self, batch_size=32, max_seq_length=640, dimensions=4096, output_mode='mean', **kwargs):
        """ Constructor for mLSTM Char Language Model Layer.

        Args:
            batch_size (int): Max size of input batches of examples, smaller batches will be padded to this size.
            max_seq_length (int): Max number of characters in input sentence, larger will be truncated.
            dimensions (int): Dimension of the hidden states of mLSTM
            output_mode (string):
                final = final hidden state of the mLSTM with shape [batch_size, dimensions]
                sequence = output every hidden state in the input sequence with shape [batch_size, max_seq_length, dimensions]
                mean = Averaged sequence output with shape [batch_size, dimensions]
        """

        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.dimensions = dimensions
        self.output_mode = output_mode.lower()
        self.weights_path = 'weights/mlstm_char_lm_weights'
        self.mlstm = None

        if self.output_mode not in ["final", "sequence", "mean"]:
            raise NameError("mLSTM output_mode (must be either final, sequence or mean but is" + self.output_mode)

        super(MLSTMCharLMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.mlstm = Model(self.weights_path, batch_size=self.batch_size, max_seq_length=self.max_seq_length)
        super(MLSTMCharLMLayer, self).build(input_shape)

    def call(self, x, **kwargs):

        # with tf.device('/device:CPU:*'): # Could optionally run with CPU if batch or seq length too large

        # Return hidden state, sequences or the mean of sequences
        # Need to reshape because py_func returns Tensors with no dimensions
        if self.output_mode in ["sequence", "mean"]:
            result = tf.py_func(func=self.mlstm.cell_transform, inp=[x], Tout=tf.float32)
            result.set_shape([x.get_shape()[0], self.max_seq_length, self.dimensions])
            if self.output_mode == "mean":
                result = tf.keras.backend.mean(result, axis=1)
        elif self.output_mode == "final":
            result = tf.py_func(func=self.mlstm.transform, inp=[x], Tout=tf.float32)
            result.set_shape([x.get_shape()[0], self.dimensions])
        else:
            raise NameError("mLSTM output_mode (must be either final, sequence or mean but is" + self.output_mode)

        return result

    def compute_output_shape(self, input_shape):
        if self.output_mode == 'final' or self.output_mode == 'mean':
            return input_shape[0], self.dimensions
        elif self.output_mode == 'sequence':
            return input_shape[0], self.max_seq_length, self.dimensions

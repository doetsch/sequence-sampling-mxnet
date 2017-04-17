import numpy as np
import mxnet as mx
import bisect
import argparse

parser = argparse.ArgumentParser(description="Sequence sampling experiments on CHiME-4",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--test', default=False, action='store_true',
                    help='whether to do testing instead of training')
parser.add_argument('--model-prefix', type=str, default=None,
                    help='path to save/load model')
parser.add_argument('--load-epoch', type=int, default=0,
                    help='load from epoch')
parser.add_argument('--num-layers', type=int, default=2,
                    help='number of stacked RNN layers')
parser.add_argument('--num-hidden', type=int, default=200,
                    help='hidden layer size')
parser.add_argument('--bidirectional', type=bool, default=True,
                    help='whether to use bidirectional layers')
parser.add_argument('--gpus', type=str,
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu. ' \
                         'Increase batch size when using multiple gpus for best performance.')
parser.add_argument('--kv-store', type=str, default='device',
                    help='key-value store type')
parser.add_argument('--num-epochs', type=int, default=25,
                    help='max num of epochs')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='the optimizer type')
parser.add_argument('--mom', type=float, default=0.0,
                    help='momentum for sgd')
parser.add_argument('--wd', type=float, default=0.00001,
                    help='weight decay for sgd')
parser.add_argument('--batch-size', type=int, default=32,
                    help='the batch size.')
parser.add_argument('--disp-batches', type=int, default=50,
                    help='show progress for every n batches')
# When training a deep, complex model, it's recommended to stack fused RNN cells (one
# layer per cell) together instead of one with all layers. The reason is that fused RNN
# cells doesn't set gradients to be ready until the computation for the entire layer is
# completed. Breaking a multi-layer fused RNN cell into several one-layer ones allows
# gradients to be processed ealier. This reduces communication overhead, especially with
# multiple GPUs.
parser.add_argument('--stack-rnn', default=False,
                    help='stack fused RNN cells to reduce communication overhead')
parser.add_argument('--dropout', type=float, default='0.0',
                    help='dropout probability (1.0 - keep probability)')

parser.add_argument('--sampling', type=str, default=None,
                    help='sequence batch sampling method: random, sorted, #partitions '
                         'in sinusoidal sampling or comma separated list of buckets')

start_label = 1
invalid_label = 0

class HDF5DATA(mx.io.DataIter):
  def one_hot(self, x):
    xs = x.reshape(x.shape[0] * x.shape[1], ) if len(x.shape) == 2 else x
    xs[xs==10429] = 0
    res = np.zeros(list(xs.shape) + [self.n_out],'int32')
    res[np.arange(xs.shape[0]), xs] = 1
    return res.reshape(x.shape[0],x.shape[1],self.n_out) if len(x.shape) == 2 else res

  def __init__(self, filename):
    self.batches = []
    h5 = h5py.File(filename, "r")
    lengths = h5["seqLengths"][...].T[0].tolist()
    xin = h5['inputs'][...]
    yin = h5['targets/data']['classes'][...]
    yin[yin==10429] = 0
    self.n_out = h5['targets/size'].attrs['classes']
    self.n_in = xin.shape[1]
    self.n_seqs = len(lengths)
    i = 0
    while i < len(lengths):
      end = min(i+BATCH_SIZE,len(lengths))
      batch_x = np.zeros((MAX_LEN, BATCH_SIZE, xin.shape[1]), 'float32')
      batch_y = np.zeros((MAX_LEN, BATCH_SIZE), 'int8')
      #batch_y = np.zeros((BATCH_SIZE, MAX_LEN), 'int32')
      batch_i = np.zeros((BATCH_SIZE, MAX_LEN), 'int8')
      for j in xrange(end-i):
        batch_x[:lengths[i+j],j] = (xin[sum(lengths[:i+j]):sum(lengths[:i+j+1])])
        #batch_y[j,:lengths[i+j]] = self.one_hot(yin[sum(lengths[:i+j]):sum(lengths[:i+j+1])])
        batch_y[:lengths[i+j],j] = yin[sum(lengths[:i+j]):sum(lengths[:i+j+1])]
        #batch_y[j * MAX_LEN:j * MAX_LEN + lengths[i+j]] = yin[sum(lengths[:i+j]):sum(lengths[:i+j+1])]
        batch_i[j,:lengths[i+j]] = 1
      self.batches.append((batch_x,batch_y,batch_i,MAX_LEN)) #max(lengths[i:end])))
      i = end
    self.lengths = lengths
    h5.close()
    self.batch_idx = 0

  def next(self):
    if self.batch_idx == len(self.batches):
      self.batch_idx = 0
      return None, None, None, None
    self.batch_idx += 1
    return self.batches[self.batch_idx-1]

  def __iter__(self):
    return self

class UtteranceIter(DataIter):
  """A data iterator for acoustic modeling of frame-wise labeled data.
  The iterator supports bucketing based on predefined bucket sizes,
  sorted and random sequence sampling, and sinusoidal sampling.  

  Parameters
  ----------
  utterances : list of list of list of floats
      utterance feature vectors with #sequences as major
  batch_size : int
      number of sequences per batch
  sampling : str, int, or list of int
      either 'sorted' or 'random' 
      or number of bins in sinusoidal sampling 
      or size of data buckets (automatically generated if None).
  layout : str
      format of data and label. 'NT' means (batch_size, length)
      and 'TN' means (length, batch_size).
  """

  def __init__(self, utterances, states, batch_size, sampling, data_name='data', label_name='labels'):
    super(UtteranceIter, self).__init__()
    if sampling is None:
      sampling = [i for i, j in enumerate(np.bincount([len(x) for x in utterances]))
                 if j >= batch_size]

    self.idx = []
    if isinstance(sampling, list):
      sampling.sort()
      max_len = max([len(utt) for utt in utterances])
      sampling.append(max_len)

      ndiscard = 0
      self.data = [[] for _ in sampling] + [[]] # last one for final bucket
      self.labels = [[] for _ in sampling] + [[]]  # last one for final bucket
      for utt, lab in zip(utterances, states):
        buck = bisect.bisect_left(sampling, len(utt))
        if buck == len(sampling):
          ndiscard += 1
          continue
        xin = np.full((sampling[sampling],len(utt[0])), 0, dtype='float32')
        xin[:len(utt)] = utt
        yout = np.full((sampling[sampling],), 0, dtype='int32')
        yout[:len(utt)] = lab
        self.data[buck].append(xin)
        self.labels[buck].append(yout)

      print("WARNING: discarded %d sentences longer than the largest bucket." % ndiscard)

      for i, buck in enumerate(self.data):
        self.idx.extend([(i, j) for j in range(0, len(buck) - batch_size + 1, batch_size)])
    else:
      raise NotImplementedError('sampling %s not supported' % str(sampling))

    self.data = [np.asarray(i, dtype='float32') for i in self.data]
    self.labels = [np.asarray(i, dtype='int32') for i in self.labels]
    self.curr_idx = 0

    self.batch_size = batch_size
    self.sampling = sampling
    self.nddata = []
    self.ndlabel = []
    self.data_name = data_name
    self.label_name = label_name
    self.sampling = sampling

    # we assume a batch major layout
    self.provide_data = [(self.data_name, (batch_size, sampling[-1], data[0].shape[1]))]
    self.provide_label = [(self.label_name, (batch_size, sampling[-1]))]

    self.reset()

  def reset(self):
    self.curr_idx = 0

    if isinstance(self.sampling, list):
      random.shuffle(self.idx) # shuffle bucket index
      for buck in self.data: # shuffle sequence index within bucket
        np.random.shuffle(buck)

      self.nddata = []
      self.ndlabel = []
      for buck_utt,buck_lab in zip(self.data,self.labels):
        self.nddata.append(ndarray.array(buck_utt, dtype='float32'))
        self.ndlabel.append(ndarray.array(buck_lab, dtype='int32'))

  def next(self):
    if self.curr_idx == len(self.idx):
      raise StopIteration

    if isinstance(self.sampling, list):
      i, j = self.idx[self.curr_idx]

      data = self.nddata[i][j:j + self.batch_size]
      label = self.ndlabel[i][j:j + self.batch_size]

      return DataBatch([data], [label], pad=0,
                       bucket_key=self.buckets[i],
                       provide_data=[(self.data_name, data.shape)],
                       provide_label=[(self.label_name, label.shape)])
    self.curr_idx += 1


def read_hdf5(fname, batching='default'):
  h5 = h5py.File(filename, "r")
  lengths = h5["seqLengths"][...].T[0].tolist()
  xin = h5['inputs'][...]
  yin = h5['targets/data']['classes'][...]
  n_out = h5['targets/size'].attrs['classes']

  utterances = []
  states = []
  offset = 0
  for length in lengths:
    utterances.append(xin[offset:offset + length])
    states.append(yin[offset:offset + length])
    offset += length

  h5.close()
  return utterances, states, n_out


def get_data():
    train_x, train_y, n_out = read_hdf5('./data/train.0001')
    valid_x, valid_y, _ = read_hdf5('./data/train.0002')

    sampling = args.sampling.split(',')
    if len(sampling) > 1: # bucket sampling
      sampling = [ int(s) for s in sampling ]
    else:
      try:
        hills = int(sampling)
        sampling = hills
      except ValueError:
        pass

    data_train  = UtteranceIter(train_x, train_y, args.batch_size, sampling=sampling)
    data_val    = UtteranceIter(valid_x, valid_y, args.batch_size, sampling=sampling)
    return data_train, data_val, n_out


def train(args):
    data_train, data_val, n_out = get_data()
    if args.stack_rnn:
        cell = mx.rnn.SequentialRNNCell()
        for i in range(args.num_layers):
            cell.add(mx.rnn.FusedRNNCell(args.num_hidden, num_layers=1,
                                         mode='lstm', prefix='lstm_l%d'%i,
                                         bidirectional=args.bidirectional))
            if args.dropout > 0 and i < args.num_layers - 1:
                cell.add(mx.rnn.DropoutCell(args.dropout, prefix='lstm_d%d'%i))
    else:
        cell = mx.rnn.FusedRNNCell(args.num_hidden, num_layers=args.num_layers, dropout=args.dropout,
                                   mode='lstm', bidirectional=args.bidirectional)

    def sym_gen(seq_len):
        data = mx.sym.Variable('data')
        label = mx.sym.Variable('labels')

        output, _ = cell.unroll(seq_len, inputs=data, merge_outputs=True, layout='NTC')
        pred = mx.sym.Reshape(output, shape=(-1, args.num_hidden*(1+args.bidirectional)))
        pred = mx.sym.FullyConnected(data=pred, num_hidden=n_out, name='pred')

        label = mx.sym.Reshape(label, shape=(-1,))
        pred = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

        return pred, ('data',), ('label',)

    if args.gpus:
        contexts = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    else:
        contexts = mx.cpu(0)

    model = mx.mod.BucketingModule(
        sym_gen             = sym_gen,
        default_bucket_key  = data_train.default_bucket_key,
        context             = contexts)

    if args.load_epoch:
        _, arg_params, aux_params = mx.rnn.load_rnn_checkpoint(
            cell, args.model_prefix, args.load_epoch)
    else:
        arg_params = None
        aux_params = None

    opt_params = {
      'learning_rate': args.lr,
      'wd': args.wd
    }

    if args.optimizer not in ['adadelta', 'adagrad', 'adam', 'rmsprop']:
        opt_params['momentum'] = args.mom

    model.fit(
        train_data          = data_train,
        eval_data           = data_val,
        eval_metric         = mx.metric.Perplexity(invalid_label),
        kvstore             = args.kv_store,
        optimizer           = args.optimizer,
        optimizer_params    = opt_params, 
        initializer         = mx.init.Xavier(factor_type="in", magnitude=2.34),
        arg_params          = arg_params,
        aux_params          = aux_params,
        begin_epoch         = args.load_epoch,
        num_epoch           = args.num_epochs,
        batch_end_callback  = mx.callback.Speedometer(args.batch_size, args.disp_batches),
        epoch_end_callback  = mx.rnn.do_rnn_checkpoint(cell, args.model_prefix, 1)
                              if args.model_prefix else None)

def test(args):
    assert args.model_prefix, "Must specifiy path to load from"
    _, data_val, vocab = get_data('NT')

    if not args.stack_rnn:
        stack = mx.rnn.FusedRNNCell(args.num_hidden, num_layers=args.num_layers,
                mode='lstm', bidirectional=args.bidirectional).unfuse()
    else:
        stack = mx.rnn.SequentialRNNCell()
        for i in range(args.num_layers):
            cell = mx.rnn.LSTMCell(num_hidden=args.num_hidden, prefix='lstm_%dl0_'%i)
            if args.bidirectional:
                cell = mx.rnn.BidirectionalCell(
                        cell,
                        mx.rnn.LSTMCell(num_hidden=args.num_hidden, prefix='lstm_%dr0_'%i),
                        output_prefix='bi_lstm_%d'%i)
            stack.add(cell)

    def sym_gen(seq_len):
        data = mx.sym.Variable('data')
        label = mx.sym.Variable('softmax_label')
        embed = mx.sym.Embedding(data=data, input_dim=len(vocab),
                                 output_dim=args.num_embed, name='embed')

        stack.reset()
        outputs, states = stack.unroll(seq_len, inputs=embed, merge_outputs=True)

        pred = mx.sym.Reshape(outputs,
                shape=(-1, args.num_hidden*(1+args.bidirectional)))
        pred = mx.sym.FullyConnected(data=pred, num_hidden=len(vocab), name='pred')

        label = mx.sym.Reshape(label, shape=(-1,))
        pred = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

        return pred, ('data',), ('softmax_label',)

    if args.gpus:
        contexts = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    else:
        contexts = mx.cpu(0)

    model = mx.mod.BucketingModule(
        sym_gen             = sym_gen,
        default_bucket_key  = data_val.default_bucket_key,
        context             = contexts)
    model.bind(data_val.provide_data, data_val.provide_label, for_training=False)

    # note here we load using SequentialRNNCell instead of FusedRNNCell.
    _, arg_params, aux_params = mx.rnn.load_rnn_checkpoint(stack, args.model_prefix, args.load_epoch)
    model.set_params(arg_params, aux_params)

    model.score(data_val, mx.metric.Perplexity(invalid_label),
                batch_end_callback=mx.callback.Speedometer(args.batch_size, 5))

if __name__ == '__main__':
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    args = parser.parse_args()

    if args.num_layers >= 4 and len(args.gpus.split(',')) >= 4 and not args.stack_rnn:
        print('WARNING: stack-rnn is recommended to train complex model on multiple GPUs')

    if args.test:
        # Demonstrates how to load a model trained with CuDNN RNN and predict
        # with non-fused MXNet symbol
        test(args)
    else:
        train(args)

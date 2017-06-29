class FullyConnectedLayer(object):

    def __init__(self,n_in, n_out, activation_fn = sigmoid, p_dropout= 0.0):
        self.n_in = n_in;

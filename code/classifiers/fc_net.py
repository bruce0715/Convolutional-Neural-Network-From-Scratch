from builtins import range
from builtins import object
import numpy as np

from stats232a.layers import *
from stats232a.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be input - fc - relu - fc - softmax.
    
    The outputs of the second fully-connected layer are the scores for each class.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=1*28*28, hidden_dim=100, num_classes=10, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Following  #
        # the instruction on class to initialize weights carefully, and biases     #
        # should be initialized to zero. All weights and biases should be stored   #
        # in the dictionary self.params, with first layer weights and biases using #
        # the keys 'W1' and 'b1' and second layer weights and biases using the     #
        # keys 'W2' and 'b2'.                                                      # 
        ############################################################################
        
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.num_classes=num_classes
        self.reg=reg

        sigma_fc=1/self.input_dim
        sigma_relu=2/self.hidden_dim

        w1=sigma_fc*np.random.randn(self.input_dim,self.hidden_dim)
        b1=np.zeros(self.hidden_dim)
        w2=sigma_relu*np.random.randn(self.hidden_dim,self.num_classes)
        b2=np.zeros(self.num_classes)

        self.params.update({'W1':w1,'b1':b1,'W2':w2,'b2':b2})

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        W1=self.params['W1']
        b1=self.params['b1']
        W2=self.params['W2']
        b2=self.params['b2']
        #reshape input
        X=X.reshape(X.shape[0],-1)
        #set forward part of network
        out_hidden,cache_hidden=fc_relu_forward(X,W1,b1)
        scores,cache_s=fc_forward(out_hidden,W2,b2)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss_scores,dx_scores=softmax_loss(scores,y)
        loss=loss_scores+0.5*self.reg*(np.sum(W1**2)+np.sum(W2**2))

        #BP
        dx2,dW2,db2=fc_backward(dx_scores,cache_s)
        dW2=dW2+self.reg*W2

        dx1,dW1,db1=fc_relu_backward(dx2,cache_hidden)
        dW1=dW1+self.reg*W1

        grads.update({'W1':dW1,'b1':db1,'W2':dW2,'b2':db2})
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {fc - [batch norm] - relu - [dropout]} x (L - 1) - fc - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=1*28*28, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        self.input_dim=input_dim
        self.num_classes=num_classes
        self.weight_scale=weight_scale

        total_dim=[self.input_dim]+hidden_dims+[self.num_classes]

        for i in range(len(total_dim)-1):
            # if i==0:
            #     Wi={'W1':(self.weight_scale/total_dim[i])*np.random.randn(total_dim[i],total_dim[i+1])}
            # else:
            #     Wi={'W'+str(i+1):(2*self.weight_scale/total_dim[i])*np.random.randn(total_dim[i],total_dim[i+1])}
            Wi={'W'+str(i+1):self.weight_scale*np.random.randn(total_dim[i],total_dim[i+1])}
            self.params.update(Wi)

        for i in range(len(total_dim)-1):
            bi={'b'+str(i+1):np.zeros(total_dim[i+1])}
            self.params.update(bi)
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        #reshape X
        X=X.reshape(X.shape[0],-1)
        layers={}
        layers['l0']=X

        for i in range(self.num_layers):
            W=self.params['W'+str(i+1)]
            b=self.params['b'+str(i+1)]
            l=layers['l'+str(i)]

            if i!=(self.num_layers-1):
                out_hidden,cache_hidden=fc_relu_forward(l,W,b)
                layers['l'+str(i+1)]=out_hidden
                layers['cache'+str(i+1)]=cache_hidden
            else: #special case of last layer
                scores,cache_s=fc_forward(l,W,b)
                layers['l'+str(i+1)]=scores
                layers['cache'+str(i+1)]=cache_s
        scores=layers['l'+str(self.num_layers)]

        #print(layers.keys())
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss_scores,dx_scores=softmax_loss(scores,y)
        
        w=[]
        for p in self.params.keys():
            if p[0]=='W':
                w_norm=np.sum(self.params[p]**2)
                w.append(w_norm)
        l2_loss=0.5*self.reg*np.sum(i for i in w)
        loss=loss_scores+l2_loss

        #BP
        layers['d'+str(self.num_layers)]=dx_scores #d3
        for i in reversed(range(self.num_layers)):
            dl=layers['d'+str(i+1)] 
            cache=layers['cache'+str(1+i)]
            
            if i==(self.num_layers-1):
                dl,dw,db=fc_backward(dl,cache)
                layers['d'+str(i)]=dl#d2
                layers['dW'+str(i+1)]=dw #dW3
                layers['db'+str(i+1)]=db #db3
            else:
                dl,dw,db=fc_relu_backward(dl,cache)
                layers['d'+str(i)]=dl #d1,d0
                layers['dW'+str(i+1)]=dw #dW1,dW2 
                layers['db'+str(i+1)]=db #db1,db2
        grads={}
        dw_dic={}
        for k,v in layers.items():
            if k[:2]=='dW': #first two characters
                dtemp={k[1:]:v+(self.reg)*self.params[k[1:]]}  #Wi
                dw_dic.update(dtemp)

        db_dic={}
        for k,v in layers.items():
            if k[:2]=='db':
                dtemp={k[1:]:v}
                db_dic.update(dtemp)

        grads.update(dw_dic)
        grads.update(db_dic)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

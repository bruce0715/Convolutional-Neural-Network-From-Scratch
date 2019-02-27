pass
from stats232a.layers import *


def fc_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    out, cache = None, None
    
    ###########################################################################
    # TODO: Implement fc-relu forward pass.                                   #
    ###########################################################################
    x_sh=x.reshape(x.shape[0],-1)
    out=np.maximum(0,(np.matmul(x_sh,w)+b))
    cache=(x,w,b)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return out, cache


def fc_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    dx, dw, db = None, None, None
    
    ###########################################################################
    # TODO: Implement the fc-relu backward pass.                              #
    ###########################################################################
    x,w,b=cache
    x_sh=x.reshape(x.shape[0],-1)
    out=np.maximum(0,(np.matmul(x_sh,w)+b))
    out_ind=np.where(out>0,1,0)
    dout2=out_ind*dout

    dx=np.matmul(dout2,np.transpose(w))
    dw=np.matmul(np.transpose(x_sh),dout2)
    db=np.matmul(np.ones(dout2.shape[0]),dout2)

    dx=dx.reshape(x.shape)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return dx, dw, db

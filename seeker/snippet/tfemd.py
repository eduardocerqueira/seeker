#date: 2022-06-06T16:59:37Z
#url: https://api.github.com/gists/466f5fb730dbf0f90d6857c31c6173dd
#owner: https://api.github.com/users/jacksonloper

import tensorflow as tf

@tf.function(input_signature=(
             tf.TensorSpec((None,None),dtype=tf.float32),
             tf.TensorSpec((None,),dtype=tf.float32),
             tf.TensorSpec((None,),dtype=tf.float32),
             tf.TensorSpec((),dtype=tf.float32),
             tf.TensorSpec((),dtype=tf.float32),
             tf.TensorSpec((),dtype=tf.float32),
             tf.TensorSpec((),dtype=tf.int32),
))
def regularized_emd_transport_inner(C,u,v,n,m,eps,niter):
    for i in tf.range(niter):
        u= - tf.math.reduce_logsumexp(-C*eps + v[None,:],axis=1) - tf.math.log(n)
        v= - tf.math.reduce_logsumexp(-C*eps + u[:,None],axis=0) - tf.math.log(m)
    return u,v

@tf.function(input_signature=(
             tf.TensorSpec((None,None),dtype=tf.float32),
             tf.TensorSpec((None,None),dtype=tf.float32),
             tf.TensorSpec((),dtype=tf.int32),
             tf.TensorSpec((),dtype=tf.float32),
))
def regularized_emd_transport(ptsA,ptsB,niter,eps):
    '''
    Abstractly,
    
        C = -||ptsA-ptsB||_1
        X = exp(-C*eps)
        for i in range(niter):
            X=X/(X.shape[0]*sum(X,axis=1,keepdims=True))
            X=X/(X.shape[1]*sum(X,axis=0,keepdims=True))
            
    returns C,u,v such that X=exp(-C*eps + u[:,None] + v[None,:])
    '''
    
    C=tf.reduce_sum(tf.math.abs(ptsA[:,None] - ptsB[None,:]),axis=-1)
    u=tf.zeros(tf.shape(ptsA)[0],dtype=ptsA.dtype)
    v=tf.zeros(tf.shape(ptsB)[0],dtype=ptsB.dtype)
     
    n=tf.cast(tf.shape(C)[0],dtype=C.dtype)
    m=tf.cast(tf.shape(C)[1],dtype=C.dtype)
    
    u,v=regularized_emd_transport_inner(C,u,v,n,m,eps,niter)
    return C,u,v
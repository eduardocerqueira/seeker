#date: 2023-12-06T16:53:46Z
#url: https://api.github.com/gists/c26811ce7be45564c382f3e4f9df6436
#owner: https://api.github.com/users/shubhamwagh

shape = (3,4,5)
a = tf.random.uniform(shape)
a_t = tf.transpose(a,(1,0,2)) # permuting first and second axis
a_concat = tf.concat([tf.reshape(a[i:i+1,:,:],(shape[1],1,shape[2])) for i in range(shape[0])],axis=1)
tf.debugging.assert_equal(a_t,a_concat)
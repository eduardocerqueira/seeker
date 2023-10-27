#date: 2023-10-27T16:52:10Z
#url: https://api.github.com/gists/65cdaae7bcd69cf431789834f5e30b72
#owner: https://api.github.com/users/Avinash07Nayak

class Decoder(tf.keras.Model):
    
    '''
    This Class takes the target sequence, encoder output sequence,
    and hidden and cell states as input and returns the predicted target data
    '''
    
    def __init__(self,vocab_size, embedding_dim, emb_matrix, lstm_units, att_units):
        
        '''
        This function initializes the variables and objects to be used in the class
        '''
        
        super().__init__()
        self.vocab_size=vocab_size
        self.embedding_dim=embedding_dim
        self.lstm_units=lstm_units
        self.emb_matrix=emb_matrix
        self.att_units=att_units
        self.onestepdecoder=OneStepDecoder(self.vocab_size, self.embedding_dim, self.emb_matrix, self.lstm_units, self.att_units)
        
    def call(self, input_to_decoder,encoder_output,decoder_hidden_state,decoder_cell_state):
        
        '''
        This function takes target sequence, output of encoder, encoder's hidden and cell states as input, 
        passes them to OneStepDecoder function and returns the target sequence as output
        '''
        
        out=tf.TensorArray(tf.float32, size=54)
        for i in range(54):
            final_output, decoder_hidden, decoder_cell, wts = self.onestepdecoder(input_to_decoder[:,i:i+1], encoder_output, decoder_hidden_state, decoder_cell_state)
            decoder_hidden_state=decoder_hidden
            decoder_cell_state=decoder_cell
            out=out.write(i, final_output)
        out = tf.transpose(out.stack(), [1, 0, 2])
        
        return out

    def get_config(self):
        
        '''
        get_config method saves the initialization arguments
        '''

        config = super().get_config()
        config['vocab_size']=self.vocab_size
        config['embedding_dim']=self.embedding_dim
        config['lstm_units']=self.lstm_units
        config['emb_matrix']=self.emb_matrix
        config['att_units']=self.att_units
        config['onestepdecoder']=self.onestepdecoder
        
        return config
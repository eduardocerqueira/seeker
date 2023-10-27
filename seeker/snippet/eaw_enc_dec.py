#date: 2023-10-27T16:56:30Z
#url: https://api.github.com/gists/30645a4e4009b4ea2d58694fae3c196b
#owner: https://api.github.com/users/Avinash07Nayak

class encoder_decoder(tf.keras.Model):
    
    '''
    This class takes the input to encoder and input to decoder data as input and returns the decoder output
    '''
    
    def __init__(self, vocab_size, embedding_dim, lstm_size, emb_matrix, att_units, btc_sz):
        
        '''
        This function initializes the variables and objects to be used in the class
        '''
        
        super().__init__()
        self.encoder = Encoder(vocab_size, embedding_dim, emb_matrix, lstm_size)
        self.decoder = Decoder(vocab_size, embedding_dim, emb_matrix, lstm_size, att_units)
        self.batch_size = btc_sz
        
    def call(self, data):
        
        '''
        This function takes the input to encoder and input to decoder data as input, passes
        encoder input to encoder, then passes the decoder input, along with encoder output 
        and its hidden and cell states to decoder to get the final output
        '''
        
        inp = data[0]
        out = data[1]
        encoder_output, hidden_state, cell_state = self.encoder(inp)#, states)
        decoder_output = self.decoder(out, encoder_output, hidden_state, cell_state)
        
        return decoder_output

    def get_config(self):
        
        '''
        get_config method saves the initialization arguments
        '''

        config = super().get_config()
        config['encoder']=self.encoder
        config['decoder']=self.decoder
        config['batch_size']=self.batch_size
        
        return config
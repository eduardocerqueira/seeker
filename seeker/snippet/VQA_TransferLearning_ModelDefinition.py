#date: 2022-05-04T17:13:01Z
#url: https://api.github.com/gists/417d656f2d24e53b5c4a98ee0901f941
#owner: https://api.github.com/users/rajannoel

max_length=24
vocab_size=embedding_matrix_train.shape[0]

def build_model(img_tensor_model):
    """
    Model definition

    Input:
      img_tensor_model: Desired pre-trained model to be used for image featurization

    Returns:
      model: Model definition
    """
    
    #IMAGE MODEL
    # im_input = Input(shape=(1280,), name = "im_input")
    if img_tensor_model == 'VGG19':
      im_input = tf.keras.layers.Input(shape=(4096))
    elif img_tensor_model == 'EfficientNet':
      im_input = tf.keras.layers.Input(shape=(1280))
    else:
      im_input = tf.keras.layers.Input(shape=(2048))
    #flat = Flatten()(im_input)
    image_model=Dense(1024,activation='relu',kernel_initializer=tf.keras.initializers.he_normal(seed=42))(im_input)

    ques_input = Input(shape=(max_length,), name = "ques_input")
    e1 =tf.keras.layers.Embedding(vocab_size, 300, weights=[embedding_matrix_train], input_length=max_length,trainable=False)(ques_input)
    l1= tf.keras.layers.LSTM(1024,kernel_initializer=tf.keras.initializers.he_normal(seed=42),return_sequences=True)(e1)
    l2= tf.keras.layers.LSTM(1024,kernel_initializer=tf.keras.initializers.he_normal(seed=42),return_sequences=True)(l1)
    
    f1= Flatten(name='flatten_1')(l2)
    question_model=Dense(1024,activation='relu',kernel_initializer=tf.keras.initializers.he_normal(seed=42))(f1)

    
    input_model=Multiply()([image_model,question_model])
    d1=BatchNormalization()(input_model)
    d1 = Dropout(0.5)(d1)
    d1=Dense(1000,activation='relu',kernel_initializer=tf.keras.initializers.he_normal(seed=42))(d1)
    final_output = Dense(901, kernel_initializer=tf.keras.initializers.he_normal(seed=42),activation='softmax')(d1)
    final_model = Model(inputs=[im_input,ques_input], outputs=final_output)
    return final_model
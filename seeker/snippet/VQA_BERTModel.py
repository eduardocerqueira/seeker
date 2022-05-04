#date: 2022-05-04T17:22:42Z
#url: https://api.github.com/gists/d4ddba9a80a5462e57417c8edc837494
#owner: https://api.github.com/users/rajannoel

# Model definitions

train_text_sbert = sbert_model.encode(df_train['question'].values) # SBert based sentence vectors
val_text_sbert = sbert_model.encode(df_val['question'].values)


def build_Xception_BERT_model():
    """
    Model definition

    Input:
      img_tensor_model: Desired pre-trained model to be used for image featurization

    Returns:
      model: Model definition
    """
    
    #IMAGE MODEL
    im_input = tf.keras.layers.Input(shape=(2048))
    #flat = Flatten()(im_input)
    image_model=Dense(1024,activation='relu',kernel_initializer=tf.keras.initializers.he_normal(seed=42))(im_input)

    ques_input = Input(shape=(768), name = "ques_input")
    question_model=Dense(1024,activation='relu',kernel_initializer=tf.keras.initializers.he_normal(seed=42))(ques_input)

    
    input_model=tf.keras.layers.Multiply()([image_model,question_model])
    d1=BatchNormalization()(input_model)
    d1 = Dropout(0.5)(d1)
    d1=Dense(1000,activation='relu',kernel_initializer=tf.keras.initializers.he_normal(seed=42))(d1)
    final_output = Dense(901, kernel_initializer=tf.keras.initializers.he_normal(seed=42),activation='softmax')(d1)
    final_model = Model(inputs=[im_input,ques_input], outputs=final_output)
    return final_model
    
# Model fitting using Xception pretrained model based image tensors and BERT embeddings based question text tensors

for img_tensor_model in ['Xception']:
  new_model, train_dataset, val_dataset = None, None, None
  tf.keras.backend.clear_session()
  new_model = build_Xception_BERT_model()
  gc.collect()
  new_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])   
  train_dataset = dataset_pipeline(train_image_path, train_text_sbert, answ_train,img_tensor_model,df_train.shape[0])
  val_dataset = dataset_pipeline(val_image_path, val_text_sbert, answ_test,img_tensor_model,df_val.shape[0])
  new_model.fit(train_dataset, epochs=20, verbose=1,workers=-1, use_multiprocessing=-1,validation_data=val_dataset, callbacks = callBacksList(),)
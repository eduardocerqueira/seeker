#date: 2022-05-04T17:14:31Z
#url: https://api.github.com/gists/6f3111d4c785afb53613eae59d28c437
#owner: https://api.github.com/users/rajannoel

for img_tensor_model in ['Xception','VGG19','ResNet50', 'EfficientNet']:
  new_model, train_dataset, val_dataset = None, None, None
  tf.keras.backend.clear_session()
  new_model = build_model(img_tensor_model)
  gc.collect()
  new_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])   
  train_dataset = dataset_pipeline(train_image_path, ques_train, answ_train,img_tensor_model,df_train.shape[0])
  val_dataset = dataset_pipeline(val_image_path, ques_test, answ_test,img_tensor_model,df_val.shape[0])
  new_model.fit(train_dataset, epochs=20, verbose=1,workers=-1, use_multiprocessing=-1,validation_data=val_dataset, callbacks = callBacksList(img_tensor_model),)
#date: 2022-07-29T17:05:10Z
#url: https://api.github.com/gists/c966155e1bbff04620f7117927f625af
#owner: https://api.github.com/users/seabnavin19


history = model.fit(
  training_set,
  validation_data=validation_set,
  epochs=30,
  callbacks=[early_stopping],
  #  class_weight={0:1.5,1:1,2:1,3:2},
  steps_per_epoch=len(training_set),
  validation_steps=len(validation_set)
)
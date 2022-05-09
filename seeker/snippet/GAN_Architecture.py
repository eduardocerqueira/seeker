#date: 2022-05-09T17:11:56Z
#url: https://api.github.com/gists/07593d677dbceaff630a5928e7e22d95
#owner: https://api.github.com/users/Shivam-316

class GAN(tf.keras.Model):
  def __init__(self, generator, discriminator, **kwargs):
    super(GAN, self).__init__(**kwargs)
    self.generator = generator
    self.discriminator = discriminator
  
  def compile(self, generator_optimizer, discriminator_optimizer, loss_fn, metric_fn):
    super(GAN, self).compile()
    self.generator_optimizer = generator_optimizer
    self.discriminator_optimizer = discriminator_optimizer
    self.loss_fn = loss_fn
    self.metric_fn = metric_fn
  
  def call(self, input, training = False):
    fake_tar_img = self.generator(input, training = training)
    return fake_tar_img
  
  def train_step(self, images):
    input_img, target_img = images
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      fake_tar_img = self(input_img, training = True)
      disc_real_output = self.discriminator([input_img, target_img], training = True)
      disc_fake_output = self.discriminator([input_img, fake_tar_img], training = True)

      total_gen_loss, gan_loss, struct_loss = self.loss_fn['generator_loss'](disc_fake_output, fake_tar_img, target_img)
      disc_loss = self.loss_fn['discriminator_loss'](disc_real_output, disc_fake_output)

    generator_gradients = gen_tape.gradient(total_gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    self.generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    self.metric_fn['total_gen_loss_tracker'].update_state(total_gen_loss)
    self.metric_fn['disc_loss_tracker'].update_state(disc_loss)

    return {
        'Avg_Total_Gen_Loss': self.metric_fn['total_gen_loss_tracker'].result(),
        'Disc_Loss': self.metric_fn['disc_loss_tracker'].result(),
    }
  
  def test_step(self, images):
    input_img, target_img = images
    fake_tar_img = self(input_img, training = True)

    disc_real_output = self.discriminator([input_img, target_img], training = True)
    disc_fake_output = self.discriminator([input_img, fake_tar_img], training = True)

    total_gen_loss, gan_loss, struct_loss = self.loss_fn['generator_loss'](disc_fake_output, fake_tar_img, target_img)
    disc_loss = self.loss_fn['discriminator_loss'](disc_real_output, disc_fake_output)

    self.metric_fn['total_gen_loss_tracker'].update_state(total_gen_loss)
    self.metric_fn['disc_loss_tracker'].update_state(disc_loss)

    return {
        'Avg_Total_Gen_Loss': self.metric_fn['total_gen_loss_tracker'].result(),
        'Disc_Loss': self.metric_fn['disc_loss_tracker'].result(),
    }

  def get_config(self):
    return {'generator': self.generator, 'discriminator': self.discriminator}

  @property
  def metrics(self):
    return [self.metric_fn['total_gen_loss_tracker'], self.metric_fn['disc_loss_tracker']]
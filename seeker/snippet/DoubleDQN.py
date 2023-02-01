#date: 2023-02-01T16:59:46Z
#url: https://api.github.com/gists/8f3f3b0ea12dff3190b2742ada3ad43d
#owner: https://api.github.com/users/Deewens

class DoubleDQNAgent():
  def __init__(self, env):
    self.env = env

    # Constants
    self.NUM_ACTIONS = self.env.action_space.n
    self.DISCOUNT_FACTOR = 0.99 # Discount factor gamma used in the Q-learning update
    self.REPLAY_START_SIZE = 5000 # The agent is run for this number of steps before the training start. The resulting experience is used to populate the replay memory
    # self.REPLAY_START_SIZE = 50
    self.FINAL_EXPLORATION_STEP = 100000 # Number of frames over which the initial value of epsilon is linearly annealed to its final value. 
    self.INITIAL_EXPLORATION = 1.0 # Initial value of epsilon in Epsilon-Greedy exploration
    self.FINAL_EXPLORATION = 0.1 # Final value of epsilon in Epsilon-Greedy exploration
    self.REPLAY_MEMORY_SIZE = 100000
    self.MINIBATCH_SIZE = 32
    self.TARGET_NETWORK_UPDATE_FREQUENCY = 1000 # The frequency with which the tqrget netzork is updqted (measured in the number of parameter updates)
    self.LEARNING_RATE = 0.00025

    self.buffer = ReplayMemory(self.REPLAY_MEMORY_SIZE)

    self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE, clipnorm=1.0)
    self.loss_function = tf.keras.losses.MeanSquaredError()

    self.model = create_q_model(self.NUM_ACTIONS)
    self.target_model = create_q_model(self.NUM_ACTIONS)
    self.target_model.set_weights(self.model.get_weights())

    #self.EPSILON_INTERVAL = (1.0 -self.FINAL_EXPLORATION)
    #self.EPSILON_DECAY_FACTOR = 0.99

    # self.running_reward = 0
    # self.episode_reward_history = []

    # self.max_episodes = 10000
    # self.max_step_per_episodes = 100

  def train(self):
    epsilon = self.INITIAL_EXPLORATION

    episode_idx = 0
    timestep = 0
    while True:
      episode_idx += 1
      episode_reward = 0
      done = False

      state, info = self.env.reset()

      while not done:
        timestep += 1
        action = self.choose_action(state, epsilon)

        # Reduce epsilon  if training started
        if timestep > self.REPLAY_START_SIZE:
          epsilon -= (self.INITIAL_EXPLORATION - self.FINAL_EXPLORATION) / self.FINAL_EXPLORATION_STEP
          epsilon = max(epsilon, self.FINAL_EXPLORATION)

        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        episode_reward += reward
        self.buffer.append(Experience(state, action, reward, done, next_state))
        
        # Only train if done observing (buffer has been filled enough)
        if timestep > self.REPLAY_START_SIZE:
          states_sample, actions_sample, rewards_sample, dones_sample, next_states_sample = self.buffer.sample(self.MINIBATCH_SIZE)
          
          states_sample_tensor = tf.convert_to_tensor(states_sample)
          next_states_sample_tensor = tf.convert_to_tensor(next_states_sample)
          actions_sample_tensor = tf.convert_to_tensor(actions_sample)

          # Perform experience replay
          # Predict the target q value from the next sample sates
          target_q_values = self.target_model(next_states_sample_tensor)

          # Calculate the target q values by discounting the discount rate from the Q Value predicted by the target model
          # (1 - minibatch.done) will be 0 if this is the terminated state, and thus, won't update the q_learning target (because 0 * x = 0)
          # reduce_max get the maximum Q_value for each list of q_values returned by the target model (because we gave a batch of 32 states to the model)
          target_q_values = rewards_sample + (1 - dones_sample) * self.DISCOUNT_FACTOR * tf.reduce_max(target_q_values, axis=1)

          # Create a mask on the action stores in the sampled minibatch 
          # This allows us to only calculate the loss on the updated Q-values
          # WHAT IS A ONE_HOT Tensor?
          # A one hot tensor is a matrix representation of a categorical variable, 
          # where the matrix has a single 1 in the column corresponding to the category and all other entries are 0. 
          # In other words, a one-hot tensor is a vector of length equal to the number of categories with a single 1 in the position corresponding to the category and all other values as 0.
          masks = tf.one_hot(actions_sample_tensor, self.NUM_ACTIONS)

          # Now, we need to calculate the gardient descent, using GradientTape to record the operation made during the training of the Q Function network (main model)
          # As stated above, GradientTape just record the operation made inside it, such as model training or calculation
          with tf.GradientTape() as tape:
            # We train the main network and record the training into the tape
            q_values = self.model(states_sample_tensor)

            # Apply the masks to the Q-values to get the Q-value only for taken action from the minibatch
            masked_q_values = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            loss = self.loss_function(target_q_values, masked_q_values)

          # We can then performed the back propagation on te taped operation made while training the network
          # Backpropagation
          gradients = tape.gradient(loss, self.model.trainable_variables)
          self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

          if timestep % self.TARGET_NETWORK_UPDATE_FREQUENCY == 0:
            # update the the target network with weights from the main network
            self.target_model.set_weights(self.model.get_weights())
            print("Target network updated")

          # Save model every 10,000 iterations
          if timestep % 10000 == 0:
            print("Saving model...")
            self.model.save('pong-model.h5', overwrite=True)

          print("Timestep: {}, Epsilon: {}, Reward: {}, Loss: {}".format(timestep, epsilon, reward, loss))

      print("Episode {} finished!".format(episode_idx))
      print("Episode reward: {}".format(episode_reward))
      print("******************")

  def choose_action(self, state, epsilon):
    # Use epsilon greedy policy to select actions.
    if np.random.random() < epsilon:
      action = self.env.action_space.sample()
    else:
      # Predict action Q-values
      state_tensor = tf.convert_to_tensor(state) # Convert the state numpy array to a tensor array because tensorflow only accept tensor
      state_tensor = tf.expand_dims(state_tensor, 0) # I don't understand what this is for
      q_values = self.model(state_tensor, training=False) # Call the model to predict the Q-Value according to the passed state

      # Take best action from the returned q_values
      # tf.argmax return the index with the largest q_values
      action = tf.argmax(q_values[0]).numpy() # convert it back to np because it returns a Tensor

    return action
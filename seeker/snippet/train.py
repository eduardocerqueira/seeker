#date: 2022-01-13T16:51:58Z
#url: https://api.github.com/gists/0e1a5601d1d1426033f003b09a259178
#owner: https://api.github.com/users/arturtoshev

logging.info('########## START TRAINING LOOP ##########')

logging.info('Parameter setup')
# dataset_dir = pathlib.Path("/home/artur/data/flag_simple")
dataset_dir = pathlib.Path("/home/artur/data/flag_simple")
model_dir = dataset_dir / "model_params"
global feature; feature = "world_pos"
learning_rate = 1e-4
epochs = 2
seed = 123

logging.info('Data setup')
# train data
ds = preprocess(f"{dataset_dir}", split="train")
inputs = tf.data.make_one_shot_iterator(ds).get_next()
inputs = {k: jnp.asarray(v) for k, v in inputs.items()}  # transform one single training example w/ target|world_pos, etc. to ndarray
global senders, receivers; senders, receivers = triangles_to_edges(inputs)
init_graph = build_single_graph(inputs, is_training=True)  
#nodes.shape=(1579,12), 0...8 for node type and 9-11 velocity
#edge.shape=(9212,7), 0...2 for world pos. distance, 3 for its norm, 4&5 for mesh pos. distance, 6 its norm


# validation data
# ds_valid = preprocess(f"{dataset_dir}", split="valid")
    
def loss_fn(params: hk.Params, inputs: Dict, is_training, key=jax.random.PRNGKey(123456)):
    """The loss function for MESHGRAPHNETS"""
    graph = build_single_graph(inputs, is_training=is_training)
    outputs = model.apply(params, key, graph)

    cur_position = inputs[feature]
    prev_position = inputs[f"prev|{feature}"]
    target_position = inputs[f"target|{feature}"]
    target_acceleration = target_position - 2 * cur_position + prev_position
    normalizer_params = out_normalizer.init(key, inputs[feature])
    target_normalized = out_normalizer.apply(normalizer_params, target_acceleration)

    node_type = inputs["node_type"].flatten()
    error = jnp.sum((target_normalized - outputs) ** 2, axis=1)
    loss = jnp.mean(jnp.where(node_type == NodeType.NORMAL, error, 0), axis=0)
    return loss

@jax.jit
def update(params, opt_state, inputs: dict, rng_key):
    value, grads = jax.value_and_grad(loss_fn)(params=params, inputs=inputs, is_training=True, key=rng_key)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, value

logging.info('Initialize model and optimizer!')
model = hk.transform(lambda x: MeshGraphNet(output_size=3, latent_size=128, num_layers=2, message_passing_steps=15)(x))
lr_schedule = optax.exponential_decay(init_value=learning_rate, transition_steps=1, decay_rate=0.999, end_value=learning_rate / 10)
optimizer = optax.adam(lr_schedule)
rng_seq = hk.PRNGSequence(seed)
params = model.init(next(rng_seq), init_graph)
opt_state = optimizer.init(params)

# load checkpoint if path provided
checkpoint_path = False  
# checkpoint_path = "./ckpts/ckpt_10000.pkl"
if checkpoint_path:
    with open(checkpoint_path, 'rb') as f:
        params_mgraphnet = pickle.load(f)  # load file

logging.info('Start training!'); start = time()
losses = jnp.zeros((epochs*8000, 2))  # 8000 corresponds to 400000/50
# counter = 0
for epoch in range(epochs):
    for i, inputs in enumerate(ds):
        inputs = {k: jnp.asarray(v) for k, v in inputs.items()}
        valid_inputs = {k: jnp.asarray(v) for k, v in tf.data.make_one_shot_iterator(ds_valid).get_next().items()}
        params, opt_state, loss_train = update(params, opt_state, inputs, next(rng_seq))
        _, _, valid_loss_val = update(params, opt_state, valid_inputs, next(rng_seq)) # TODO: Only evaluate loss, but loss not jitted?
        if i % 50 == 0:  # log frequency
            # loss_val = loss_fn(params=params, inputs=inputs, is_training=False, key=next(rng_seq))
            # losses = losses.at[i].set(jnp.array([loss_train, loss_val]))
            # logging.info(f"epoch {epoch}/{epochs}\t iter {i}\t loss train/val: {loss_train:.6f}/{loss_val:.6f}\t valid loss: {valid_loss_val:.6f}\t ({(time()-start):.1f}sec)")
            logging.info(f"epoch {epoch}/{epochs}\t iter {i}\t loss train: {loss_train:.6f}\t valid loss: {valid_loss_val:.6f}\t ({(time()-start):.1f}sec)")
            # counter += 1
            if i % 10000 == 0:  # save checkpoint every 10k steps
                with open('./ckpts/ckpt_'+str(i)+'.pkl', 'wb') as f:
                    pickle.dump(params, f)

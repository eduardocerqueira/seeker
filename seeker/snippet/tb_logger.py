#date: 2022-11-04T17:01:54Z
#url: https://api.github.com/gists/6e4cfb4bb8468d1a51c1ef3b44482bf5
#owner: https://api.github.com/users/fdsig

from tensorboardX import SummaryWriter
import wandb
from pathlib import Path
import numpy as np
from PIL import Image
wandb.login(host='https://qa-google.wandb.io',key='***')
# defince error ints and truncated meaing
errors, error_names = [1,2,126,127,128,129,130,255], ['Catchall' , 
                                                      'Misuse of shell', 
                                                      'cannot execute',
                                                      'Invalid argument',
                                                      'Fatal',
                                                      'Control-C',
                                                      'range']
# Snippet to initialise the wandb with sync tensorboard
model_name = 'to_tensor_board'
experiment_name = 'base_board'
save_folder = Path('save_dir/')
if not save_folder.exists:
  save_folder.mkdir()
wandb_config = dict(
    project = f'{model_name}-{experiment_name}',
    name = 'testing_data_from_TB',
    entity = 'demonstrations',
)
run = wandb.init(**wandb_config, sync_tensorboard=True)
​
​
# Snippets used to log errors
tb_writer = SummaryWriter(save_folder)
​
# These lines are used wherever we want to log error terms
for step,(name,error )in enumerate(zip(error_names, errors)):
    tb_writer.add_scalar(name,error,global_step=step)
​
accuracies, losses = np.linspace(0,1,20), np.linspace(1,0,20)
# logs some dummy accuracy and loss
for step,(acc,loss) in enumerate(zip(accuracies,losses)):
  tb_writer.add_scalar('accuracy', acc, global_step=step)
  tb_writer.add_scalar('loss', loss, global_step=step)
  image = np.random.randint(low=0, high=255, size=(32, 32, 1), dtype=np.uint8)
  tb_writer.add_image(f'image {step}',image, dataformats='HWC',global_step=step)
run.finish()
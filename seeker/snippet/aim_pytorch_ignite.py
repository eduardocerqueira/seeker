#date: 2022-02-23T17:05:37Z
#url: https://api.github.com/gists/728c4c4623be89c61d9bbad0019ab8eb
#owner: https://api.github.com/users/SGevorg

# call aim sdk designed for pytorch ignite
from aim.pytorch_ignite import AimLogger

# track experimential data by using Aim
aim_logger = AimLogger(
    experiment='aim_on_pt_ignite',
    train_metric_prefix='train_',
    val_metric_prefix='val_',
    test_metric_prefix='test_',
)

# track experimential data by using Aim
aim_logger.attach_output_handler(
    train_evaluator,
    event_name=Events.EPOCH_COMPLETED,
    tag="train",
    metric_names=["nll", "accuracy"],
    global_step_transform=global_step_from_engine(trainer),
)
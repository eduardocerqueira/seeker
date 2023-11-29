#date: 2023-11-29T16:56:05Z
#url: https://api.github.com/gists/d222fe2eb7b2ecbf87d924a762c1f7d4
#owner: https://api.github.com/users/rnyak

import os
import tensorflow as tf
import merlin.models.tf as mm

from nvtabular.workflow import Workflow

from merlin.systems.dag.ops.tensorflow import PredictTensorflow
from merlin.systems.dag.ensemble import Ensemble
from merlin.systems.dag.ops.workflow import TransformWorkflow

loaded_model = tf.keras.models.load_model('/workspace/data/saved_model')


wf = Workflow.load("/workspace/data/workflow_etl")
inf_ops = wf.input_schema.column_names >> TransformWorkflow(wf) >> PredictTensorflow(loaded_model)
ensemble = Ensemble(inf_ops, wf.input_schema)
ensemble.export(os.path.join('/workspace/data/', 'ensemble_reloaded'))
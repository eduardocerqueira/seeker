#date: 2022-01-20T17:15:12Z
#url: https://api.github.com/gists/744486f1f6e2c998cb2b0dafb57dfc8f
#owner: https://api.github.com/users/shuklak13

"""Model Card TFX Component.

The ModelCardGenerator is used to generate model cards in TFX pipelines.
"""

from typing import Any, List, Tuple, Optional

from model_card_toolkit.tfx import artifact
from model_card_toolkit.tfx import executor

from tfx import types
from tfx.dsl.components.base import executor_spec
from tfx.dsl.components.base.base_component import BaseComponent
from tfx.types import component_spec
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs

MODEL_CARD_KEY = 'model_card'


class ModelCardComponentSpec(component_spec.ComponentSpec):
  """Component spec for Model Card TFX component."""
  PARAMETERS = {
      'json':
          component_spec.ExecutionParameter(type=str, optional=True),
      # template_io's type is List[Tuple[str, str]],
      # but we need List[Any] to pass ExecutionParameter.type_check().
      # See below link for details.
      # https://github.com/tensorflow/tfx/blob/4ff5e97b09540ff8a858076a163ecdf209716324/tfx/types/component_spec.py#L308
      'template_io':
          component_spec.ExecutionParameter(
              type=List[Any], optional=True)
  }
  INPUTS = {
      standard_component_specs.STATISTICS_KEY:
          component_spec.ChannelParameter(
              type=standard_artifacts.ExampleStatistics),
      standard_component_specs.EVALUATION_KEY:
          component_spec.ChannelParameter(
              type=standard_artifacts.ModelEvaluation),
      standard_component_specs.PUSHED_MODEL_KEY:
          component_spec.ChannelParameter(type=standard_artifacts.PushedModel),
  }
  OUTPUTS = {
      MODEL_CARD_KEY:
          component_spec.ChannelParameter(type=artifact.ModelCard),
  }


class ModelCardGenerator(BaseComponent):
  """A TFX component to generate a model card.

  Uses ExampleStatistics, ModelEvaluation, and PushedModel to generate the model
  card. Writes a ModelCard.

  Accepts `json` to populate model card fields manually.

  Accepts `template_io` to use custom Jinja templates.
  """

  SPEC_CLASS = ModelCardComponentSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(self,
               statistics: Optional[types.Channel] = None,
               evaluation: Optional[types.Channel] = None,
               pushed_model: Optional[types.Channel] = None,
               json: Optional[str] = None,
               template_io: Optional[List[Tuple[str, str]]] = None
              ):
    spec = ModelCardComponentSpec(
        statistics=statistics,
        evaluation=evaluation,
        pushed_model=pushed_model,
        model_card=types.Channel(type=artifact.ModelCard),
        json=json,
        template_io=template_io)
    super(ModelCardGenerator, self).__init__(spec=spec)

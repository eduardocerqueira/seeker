#date: 2024-02-27T17:03:04Z
#url: https://api.github.com/gists/a6096429de86ef333fbdf5867d02818f
#owner: https://api.github.com/users/FabianVegaA

from collections import deque
from contextlib import contextmanager, AbstractContextManager
from typing import TypeVar, Generic

import django


Model = Generic[TypeVar('Model', bound=django.db.models.Model)]


class BulkedOperation:
    __slots__ = ('model', 'updates', 'creates')

    model: Model
    updates: deque[Model]
    creates: deque[Model]

    def __init__(self, model):
        self.model = model

        self.updates = deque()
        self.creates = deque()

    def update(self, model_object, **field_updates: dict[str, ...]) -> None:
        for field, value in field_updates.items():
            setattr(model_object, field, value)
        self.updates.append(model_object)

    def create(self, model_object, **init_fields: dict[str, ...]) -> None:
        self.creates.append(model_object(**init_fields))

    def execute(
        self,
        batch_size: int | None = None,
        **bulk_create_or_update_kwargs: dict[str, ...],
    ) -> None:
        create_options = {option for option in bulk_create_or_update_kwargs if option.startswith('create__')}
        update_options = {option for option in bulk_create_or_update_kwargs if option.startswith('update__')}
        self.model.objects.bulk_create(self.creates, batch_size=batch_size, **create_options)
        self.model.objects.bulk_update(self.updates, batch_size=batch_size, **update_options)


@contextmanager
def bulk_operations(
    model: Model,
    batch_size: int | None = None,
    **bulk_create_or_update_kwargs: dict[str, ...],
) -> AbstractContextManager['BulkedOperation[Model]']:
    """Create a bulk operation context for the given model.

    Args:
        model: The model to perform bulk operations on.
        batch_size: The batch size to use for bulk operations. If not provided, the default batch size will be used.
        **bulk_create_or_update_kwargs: Additional keyword arguments to pass to the bulk create or update methods. 
            For example, to specify the fields to update, use `update__fields=('field1', 'field2')`, and to specify
            the fields to create, use `create__fields=('field1', 'field2')`.

    >>> with bulk_operations(MyModel) as my_model_bulk:
    ...     my_model_bulk.update(obj1, field1='new_value')
    ...     my_model_bulk.create(field1='value', field2='value')
    """
    bulker: BulkedOperation[Model] = type('BulkedOperation', (Generic[Model], BulkedOperation))(model)
    yield bulker
    bulker.execute(batch_size=batch_size, **bulk_create_or_update_kwargs)

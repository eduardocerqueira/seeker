#date: 2025-04-03T16:53:29Z
#url: https://api.github.com/gists/207cb96fb653fff7735e92117805c933
#owner: https://api.github.com/users/buffmomoeveryday

from typing import Any, TypeVar, Type
from django.db import models
from pydantic import BaseModel as PydanticBaseModel

ModelType = TypeVar("ModelType", bound=models.Model)
PydanticType = TypeVar("PydanticType", bound=PydanticBaseModel)


def update_model(model: ModelType, data: PydanticType) -> ModelType:
    """
    Updates a Django model instance with data from a Pydantic object.

    Args:
        model: The Django model instance to update.
        data: The Pydantic object containing the update data.

    Returns:
        The updated Django model instance.

    Raises:
        TypeError: If the 'model' argument is not a Django model instance,
        or if the 'data' argument is not a Pydantic object.
    """
    if not isinstance(model, models.Model):
        raise TypeError("The 'model' argument must be a Django model instance.")

    if not isinstance(data, PydanticBaseModel):
        raise TypeError("The 'data' argument must be a Pydantic object.")

    for attr, value in data.dict().items():
        setattr(model, attr, value)
    return model
    
    
    
#usage
@movie_router.put(
    "/{id}/",
    response={
        200: MessageSchema,
        500: MessageSchema,
        404: MessageSchema,
    },
)
def update_movie(request, id, data: MovieInSchema):
    try:
        movie = get_object_or_404(Movie, id=id)
        movie = update_model(movie, data)
        movie.save()
        return 200, {"message": "updated successfully"}

    except Exception as e:
        return 500, {"message": f"some error occoured {str(e)}"}

    except Movie.DoesNotExist as e:
        return 404, {"message": "movie doesn't exist"}


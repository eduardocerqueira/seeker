#date: 2025-04-25T16:42:32Z
#url: https://api.github.com/gists/a9b7ba75059a2284af5058f523b8e5b4
#owner: https://api.github.com/users/thameemk


from bson import ObjectId
from pymongo.client_session import ClientSession


async def get_pymongo_session(model) -> ClientSession:
    mongo_client = model._get_db().client
    return mongo_client.start_session()


async def insert_with_session(session: ClientSession, model):
    model.validate()
    data = model.to_mongo().to_dict()
    mongo_response = model._get_collection().insert_one(data, session=session)
    data['_id'] = mongo_response.inserted_id
    return model._from_son(data)


async def find_and_update_with_session(session: ClientSession,
                                       document_id: ObjectId,
                                       new_data: dict, model):
    updated_data = model._get_collection().find_one_and_update(
        {"_id": document_id},
        {
            "$set": new_data
        },
        session=session,
        return_document=True
    )
    updated_data['_id'] = document_id
    return model._from_son(updated_data)

# Usage
mongo_session = await get_pymongo_session(SampleModel)

with mongo_session.start_transaction():
  data = await insert_with_session(mongo_session, sample_model_object)
  data = await insert_with_session(mongo_session, sample_2_model_object)

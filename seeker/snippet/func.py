#date: 2022-07-27T16:54:53Z
#url: https://api.github.com/gists/3530a2b9439e5ef4a1efcc00f7ba573b
#owner: https://api.github.com/users/davidallan

import io
import json
import logging
import pandas
import base64
from avro.io import DatumReader
from avro.datafile import DataFileReader
from avro.io import BinaryDecoder

from fdk import response

import oci
from oci.ai_language.ai_service_language_client import AIServiceLanguageClient

def handler(ctx, data: io.BytesIO=None):
    signer = oci.auth.signers.get_resource_principals_signer()
    resp = do(signer,data)
    return response.Response(
        ctx, response_data=resp,
        headers={"Content-Type": "application/json"}
    )

def nr(dip, docs):
    details = oci.ai_language.models.BatchDetectLanguageEntitiesDetails(documents=docs)
    le = dip.batch_detect_language_entities(batch_detect_language_entities_details=details)
    if len(le.data.documents) > 0:
      return json.loads(le.data.documents.__repr__())
    return {}

def do(signer, data):
    dip = AIServiceLanguageClient(config={}, signer=signer)

    body = json.loads(data.getvalue())
    input_data = base64.b64decode(body.get("data")).decode("utf-8")
    input_parameters = body.get("parameters")
    col = input_parameters.get("column")
    language_code = input_parameters.get("language_code")
    if language_code is None:
      language_code = "en"

    docs=[]
    for jsonObj in input_data.split('\n'):
      if (len(jsonObj) == 0):
        continue
      doc = eval(jsonObj)
      skey=str(doc['secret_id_field'])
      adoc = oci.ai_language.models.EntityDocument(key=skey, language_code=language_code, text=doc[col])
      docs.append(adoc)

    retdocs = nr(dip, docs)

    # each document has a list of entities
    df = pandas.DataFrame([t for t in retdocs ])

    #Explode the array of entities into row per entity
    dfe = df[df['entities'].map(len) > 0]

    dfe = dfe.explode('entities',True)
    #Add a column for each property we want to return from entity struct
    ret=pandas.concat([dfe,pandas.DataFrame((d for idx, d in dfe['entities'].iteritems()))], axis=1)

    #Drop array of entities column
    ret = ret.drop(['entities'],axis=1)

    #Drop the input text and language_code columns we don't need to return
    ret = ret.drop("language_code",axis=1)
    ret = ret.rename(columns={"key": "secret_id_field"})

    strdata=ret.to_json(orient='records')
    return strdata

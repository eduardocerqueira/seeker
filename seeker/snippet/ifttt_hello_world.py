#date: 2021-12-23T17:18:53Z
#url: https://api.github.com/gists/50d92bbc2a707a64c6867f0f3f123064
#owner: https://api.github.com/users/artrybalko

from fastapi import FastAPI, status, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from typing import Optional
from datetime import datetime, timezone
import uuid

app = FastAPI()

class TriggerCheck(BaseModel):
    limit: Optional[int]

class QueryRun(BaseModel):
    limit: Optional[int]

IFTTT_SERVICE_KEY = 'Paste your service key here (find yours at https://ift.tt/your-service-key)'

# Check the validity of the IFTTT-Service-Key on each incoming request before passing it to the path operation
@app.middleware("http")
async def check_service_key(request: Request, call_next):
    headers = request.headers
    if 'IFTTT-Service-Key' not in headers or headers['IFTTT-Service-Key'] != IFTTT_SERVICE_KEY :
      content = {"errors": [{"message": "Unauthorized"}]}
      return JSONResponse(content = content, status_code = status.HTTP_401_UNAUTHORIZED)
    response = await call_next(request)
    return response

# Generate a mock event used in trigger and query responses
def generate_event():
  event = {
        'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'meta': {
          'id': str(uuid.uuid4()),
          'timestamp': int(datetime.now(timezone.utc).timestamp())
        }
      }
  return event

# Service health check
@app.get('/ifttt/v1/status', status_code=status.HTTP_200_OK)
def check():
    return

# Test data to run subsequent tests with
@app.post('/ifttt/v1/test/setup', status_code=status.HTTP_200_OK)
def test_setup():
    data = {
      'samples': {
        'actionRecordSkipping': {
          'create_new_thing': { 'invalid': 'true' }
        }
      }
    }
    return {'data': data}

# Trigger endpoint
@app.post('/ifttt/v1/triggers/new_thing_created', status_code=status.HTTP_200_OK)
def new_thing_created(trigger_check: TriggerCheck):
  data = []
  numOfItems = trigger_check.limit
  if numOfItems is None:
    numOfItems = 3
  if numOfItems >= 1:
    i = 0
    while i < numOfItems:
      i += 1
      data.append(generate_event())
  return {'data': data}

# Query endpoint
@app.post('/ifttt/v1/queries/list_all_things', status_code=status.HTTP_200_OK)
def list_all_things(query_run: QueryRun):
  data = []
  numOfItems = query_run.limit
  if numOfItems is None:
    numOfItems = 3
  if numOfItems >= 1:
    i = 0
    while i < numOfItems:
      i += 1
      data.append(generate_event())
  cursor = None
  if query_run.limit == 1:
    cursor = str(uuid.uuid4())
  return {
    'data': data,
    'cursor': cursor
    }

# Action endpoint
@app.post('/ifttt/v1/actions/create_new_thing', status_code=status.HTTP_200_OK)
def create_new_thing():
  id = str(uuid.uuid4())
  return {'data': [{'id': id}]}
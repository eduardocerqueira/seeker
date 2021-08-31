#date: 2021-08-31T03:19:38Z
#url: https://api.github.com/gists/33ef14a8686e4c2aeacb4a52f37cb932
#owner: https://api.github.com/users/davidallan

import io
import json
import oci
from io import StringIO
import paramiko
from paramiko import Transport, SFTPClient, RSAKey
from fdk import response
import base64
from oci.object_storage import UploadManager
from oci.object_storage.transfer.constants import MEBIBYTE

# Retrieve secret
def read_secret_value(secret_client, secret_id):
    response = secret_client.get_secret_bundle(secret_id)
    base64_Secret_content = response.data.secret_bundle_content.content
    base64_secret_bytes = base64_Secret_content.encode('ascii')
    base64_message_bytes = base64.b64decode(base64_secret_bytes)
    secret_content = base64_message_bytes.decode('ascii')
    return secret_content

def handler(ctx, data: io.BytesIO = None):
    signer = oci.auth.signers.get_resource_principals_signer()

    body = json.loads(data.getvalue())
    host = body.get("host")
    username = body.get("user")
    source_file = body.get("sftp_file")
    bucket = body.get("bucket")
    object_name = body.get("object_name")
    operation = body.get("operation")
    secret_id = body.get("secret")

    if operation is None:
      operation = "PUT"

    # In the base case, configuration does not need to be provided as the region and tenancy are obtained from the InstancePrincipalsSecurityTokenSigner
    identity_client = oci.identity.IdentityClient(config={}, signer=signer)
    # Get instance principal context
    secret_client = oci.secrets.SecretsClient(config={}, signer=signer)
    secret_contents = read_secret_value(secret_client, secret_id)

    pkey = paramiko.RSAKey.from_private_key(StringIO(secret_contents))
    port = 22

    con = Transport(host, port)
    con.connect(None,username=username, pkey=pkey)
    sftp = SFTPClient.from_transport(con)
    #sftp.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    if operation == "PUT":
      src_file = None
      try:
        src_file = sftp.file(source_file)
      except IOError:
        sftp.close()
        resp_data = {"status":"Cannot find:"+source_file}
        return response.Response(
            ctx, response_data=resp_data, headers={"Content-Type": "application/json"}
        )

      object_storage_client = oci.object_storage.ObjectStorageClient(config={}, signer=signer)
      namespace = object_storage_client.get_namespace().data
      part_size = 10 * MEBIBYTE  # part size (in bytes)
      upload_manager = UploadManager(object_storage_client, allow_parallel_uploads=True, parallel_process_count=3)
      stat = upload_manager.upload_stream(
          namespace, bucket, object_name, stream_ref=src_file, part_size=part_size) #, progress_callback=progress_callback)

      sftp.close()
      resp_data = {"status":stat.status}
      return response.Response(
          ctx, response_data=resp_data, headers={"Content-Type": "application/json"}
      )
    else:
      object_storage_client = oci.object_storage.ObjectStorageClient(config={}, signer=signer)
      namespace = object_storage_client.get_namespace().data
      rslt = object_storage_client.get_object(namespace,
                            bucket,
                            object_name)
      with sftp.file(source_file, 'wb') as f:
          for chunk in rslt.data.raw.stream(1024 * 1024, decode_content=False):
              f.write(chunk)


      f.close()
      sftp.close()
      resp_data = {"status":"200"}
      return response.Response(
          ctx, response_data=resp_data, headers={"Content-Type": "application/json"}
      )

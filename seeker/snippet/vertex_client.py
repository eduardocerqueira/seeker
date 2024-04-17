#date: 2024-04-17T17:06:59Z
#url: https://api.github.com/gists/e4d711a4239e6097dc084354cb335b35
#owner: https://api.github.com/users/hugoleborso

class EmbeddingsClient:
    def __init__(self) -> None:
        self.project_id = GCP_PROJECT_ID
        self.location = GCP_REGION
        self.endpoint = (
            f"projects/{self.project_id}/locations/{self.location}"
            "/publishers/google/models/multimodalembedding@001"
        )

        self.client_options = {
            "api_endpoint": f"{self.location}-aiplatform.googleapis.com"
        }
        self.client = aiplatform.gapic.PredictionServiceClient(
            client_options=self.client_options
        )

    def get_image_embeddings(self, image_bytes: bytes) -> list[float]:
        instance = struct_pb2.Struct()
        encoded_content = base64.b64encode(image_bytes).decode("utf-8")
        image_struct = instance.fields["image"].struct_value
        image_struct.fields["bytesBase64Encoded"].string_value = encoded_content

        request = aiplatform.gapic.PredictRequest()
        request.endpoint = self.endpoint
        request.instances.append(instance)
        predictions = self.client.predict(request).predictions
        if image_bytes:
            image_emb_value = predictions[0]["imageEmbedding"]
            return [v for v in image_emb_value]
        return []
//date: 2024-06-25T16:37:49Z
//url: https://api.github.com/gists/0a74ec0ec160538c39313e73f27f8740
//owner: https://api.github.com/users/vanduc2514

public class VertexAIAdapter extends VertexAI {

    private String geminiApiKey;

    private String geminiEndpoint;

    private PredictionServiceClient predictionServiceClient;

    public VertexAIAdapter(String geminiEndpoint, String geminiApiKey) {
        // Avoid exception
        super("dummy", "dummy");
        this.geminiApiKey = geminiApiKey;
        this.geminiEndpoint = geminiEndpoint;
    }

    @Override
    public PredictionServiceClient getPredictionServiceClient() {
        if (predictionServiceClient == null) {
            try {
                PredictionServiceSettings settings = PredictionServiceSettings.newBuilder()
                    .setEndpoint(geminiEndpoint)
                    .setCredentialsProvider(new NoCredentialsProvider())
                    .setTransportChannelProvider(InstantiatingHttpJsonChannelProvider.newBuilder().build())
                    .build();
                var predictionServiceAdapter = new PredictionServiceAdapter(settings, geminiApiKey);
                predictionServiceClient = new PredictionServiceClient(predictionServiceAdapter) {};
            } catch (IOException exception) {
                throw new RuntimeException("Cannot create PredictionServiceClient");
            }
        }
        return predictionServiceClient;
    }

    private class PredictionServiceAdapter extends PredictionServiceStub {

      private static final String DEFAULT_MODEL = "gemini-1.5-pro";

      private PredictionServiceSettings predictionServiceSettings;

      private ClientContext clientContext;

      private BackgroundResource backgroundResources;

      private String geminiApiKey;

      private TypeRegistry typeRegistry;

      public PredictionServiceAdapter(
              PredictionServiceSettings predictionServiceSettings,
              String geminiApiKey) throws IOException {
          this.predictionServiceSettings = predictionServiceSettings;
          this.geminiApiKey = geminiApiKey;
          typeRegistry = TypeRegistry.newBuilder().build();
          clientContext = ClientContext.create(predictionServiceSettings);
          backgroundResources = new BackgroundResourceAggregation(clientContext.getBackgroundResources());
      }

      @Override
      public UnaryCallable<GenerateContentRequest, GenerateContentResponse> generateContentCallable() {
          HttpJsonCallSettings<GenerateContentRequest, GenerateContentResponse> httpJsonCallSettings = HttpJsonCallSettings
                  .<GenerateContentRequest, GenerateContentResponse>newBuilder()
                  .setMethodDescriptor(createApiMethodDescriptor("generateContent", ApiMethodDescriptor.MethodType.UNARY))
                  .setTypeRegistry(typeRegistry)
                  .build();
          return HttpJsonCallableFactory.createUnaryCallable(
                  httpJsonCallSettings,
                  predictionServiceSettings.generateContentSettings(),
                  clientContext);
      }

      @Override
      public ServerStreamingCallable<GenerateContentRequest, GenerateContentResponse> streamGenerateContentCallable() {
          HttpJsonCallSettings<GenerateContentRequest, GenerateContentResponse> httpJsonCallSettings = HttpJsonCallSettings
                  .<GenerateContentRequest, GenerateContentResponse>newBuilder()
                  .setMethodDescriptor(createApiMethodDescriptor("streamGenerateContent", ApiMethodDescriptor.MethodType.SERVER_STREAMING))
                  .setTypeRegistry(typeRegistry)
                  .build();
          return HttpJsonCallableFactory.createServerStreamingCallable(
                  httpJsonCallSettings,
                  predictionServiceSettings.streamGenerateContentSettings(),
                  clientContext);
      }

      private ApiMethodDescriptor<GenerateContentRequest, GenerateContentResponse> createApiMethodDescriptor(String methodName, ApiMethodDescriptor.MethodType methodType) {
          ProtoMessageRequestFormatter<GenerateContentRequest> requestFormatter =
              ProtoMessageRequestFormatter.<GenerateContentRequest>newBuilder()
                  .setPath("/v1beta/models/{model=*}:" + methodName, request -> Map.of("model", DEFAULT_MODEL))
                  .setQueryParamsExtractor(request -> Map.of("key", List.of(geminiApiKey)))
                  .setRequestBodyExtractor(request -> ProtoRestSerializer.create().toBody("*", request.toBuilder().clearModel().build(),false))
                  .build();
          ProtoMessageResponseParser<GenerateContentResponse> responseParser =
              ProtoMessageResponseParser.<GenerateContentResponse>newBuilder()
                  .setDefaultInstance(GenerateContentResponse.getDefaultInstance())
                  .setDefaultTypeRegistry(typeRegistry)
                  .build();
          return ApiMethodDescriptor.<GenerateContentRequest, GenerateContentResponse>newBuilder()
                  .setFullMethodName("google.cloud.aiplatform.v1.PredictionService/" + methodName)
                  .setHttpMethod("POST")
                  .setType(methodType)
                  .setRequestFormatter(requestFormatter)
                  .setResponseParser(responseParser)
                  .build();
      }

      @Override
      public void shutdown() {
          backgroundResources.shutdown();
      }

      @Override
      public boolean isShutdown() {
          return backgroundResources.isShutdown();
      }

      @Override
      public boolean isTerminated() {
          return backgroundResources.isTerminated();
      }

      @Override
      public void shutdownNow() {
          backgroundResources.shutdownNow();
      }

      @Override
      public boolean awaitTermination(long duration, TimeUnit unit) throws InterruptedException {
          return backgroundResources.awaitTermination(duration, unit);
      }

      @Override
      public void close() {
          try {
              backgroundResources.close();
          } catch (Exception e) {
              throw new RuntimeException(e.getMessage(), e);
          }
      }

  }

}
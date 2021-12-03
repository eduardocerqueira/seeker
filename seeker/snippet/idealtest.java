//date: 2021-12-03T16:54:02Z
//url: https://api.github.com/gists/0d5a9b656abd35200ce28e578b8b7a84
//owner: https://api.github.com/users/arjun-sd

public class RouteServiceTest {
  public static final String ORG_NAME = "acmecorp";
  public static final String SIGNADOT_API_KEY = System.getenv("SIGNADOT_API_KEY");

  public void createWorkspace() throws ApiException {
        apiClient = new ApiClient();
        apiClient.setApiKey(SIGNADOT_API_KEY);
        workspacesApi = new WorkspacesApi(apiClient);
    
        ServiceMetadata svcMD = new ServiceMetadata("cluster1", "route", "signadot/hotrod-route-staging", "latest");
        
        Workspace w = scvMD.fork("MyWs", "Sample Desc");
         
        Signadotv1CreateWorkspaceRequest request = new SignadotSingleSvcCreateWorkspaceRequest("cluster1", 
                                                                                        "MyTestWorkspace, 
                                                                                        "Testing Svc Foo", 
                                                                                        "signadot/hotrod-route-staging",
                                                                                        "latest");
        workspace = workspacesApi.createNewWorkspace(ORG_NAME, request);
            
        String previewURL = workspace.getPreviewURL();
        if (previewURL == null) {
            throw new RuntimeException("preview URL not generated");
        }

        // set the base URL for tests
        RestAssured.baseURI = previewURLs.get(0);
    }
}
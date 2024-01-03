//date: 2024-01-03T16:52:45Z
//url: https://api.github.com/gists/5e019de1a892bbdd6145877c72ccb449
//owner: https://api.github.com/users/SA-JackMax

public class NetgearJSUtils extends BaseScopableProcessorExtension {

    private static final Log logger = LogFactory.getLog(NetgearJSUtils.class);

    private NodeService nodeService;
    private SearchService searchService;
    private ServiceRegistry serviceRegistry;
    private ActionService actionService;



    public void executeAsyncAction(NodeRef nodeRef){
        Action createL4SubFoldersAction = actionService.createAction("create-l4-subfolders");
        new Thread(new ExecuteActionTask(createL4SubFoldersAction, nodeRef)).start();

    }

    public void setNodeService(NodeService nodeService) {
        this.nodeService = nodeService;
    }

    public void setSearchService(SearchService searchService) {
        this.searchService = searchService;
    }

    public void setServiceRegistry(ServiceRegistry serviceRegistry) {
        this.serviceRegistry = serviceRegistry;
    }

    public void setActionService(ActionService actionService) {
        this.actionService = actionService;
    }
}
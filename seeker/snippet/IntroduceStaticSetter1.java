//date: 2024-09-05T17:06:13Z
//url: https://api.github.com/gists/6c259d3e1c1c3d3c0e02c580cbbdf1c5
//owner: https://api.github.com/users/trikitrok

class MessageRouter {
    public void Route(Message message) {
        //!! ouch... x(
        ExternalRouter.getInstance().sendMessage(message);
    }
}

class ExternalRouter // another Singleton! x(
{
    private static ExternalRouter instance;

    private ExternalRouter() {
        // initialize stuff
    }

    public static ExternalRouter getInstance() {
        if (instance == null) {
            instance = new ExternalRouter();
        }
        return instance;
    }
    
    //!! Added for testing purposes only, do not use this in production code
    public static void setInstanceForTesting(ExternalRouter anInstance) {
        instance = anInstance;
    }

    // more code...
    public void sendMessage(Message message) {
        // interesting code to send the message
    }
}

/////////////////////////////////////////////
// In some test we use the static setter to 
// set a test double so that we can control 
// what the singleton's instance does

class MessageRouterTest
{
    @Test
    public void routes_message()
    {
        ExternalRouter externalRouter = mock(ExternalRouter.class);
        ExternalRouter.setInstanceForTesting(externalRouter);
        MessageRouter messageRouter = new MessageRouter();
        Message message = new Message();
        
        messageRouter.Route(message);
        
        verify(externalRouter).sendMessage(message);
    }

    // some other tests...

   @AfterEach
    public void TearDown()
    {
        // to keep tests isolated
        ExternalRouter.setInstanceForTesting(null);
    }
}
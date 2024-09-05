//date: 2024-09-05T16:52:22Z
//url: https://api.github.com/gists/b9c886d5f45c4da84e1e28746c104b79
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

    // more code...
    public void sendMessage(Message message) {
        // interesting code to send the message
    }
}
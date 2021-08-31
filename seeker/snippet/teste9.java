//date: 2021-08-31T02:23:42Z
//url: https://api.github.com/gists/2d89465946a200285062d1b6dae843ac
//owner: https://api.github.com/users/letpires

// CERTO
public class FitNesseServer implements SocketServer { 
    private FitNesseContext context; 

public FitNesseServer (FitNesseContext context) {
    this.context = context; 
} 

public void serve(Socket s) {
    serve(s, 10000); 
} 
  
public void serve(Socket s, long requestTimeout) { 
    try { 
      FitNesseExpediter sender = new FitNesseExpediter(s, context); 
      sender.setRequestParsingTimeLimit(requestTimeout); 
      sender.start(); 
    } 
    catch(Exception e) { 
      e.printStackTrace(); 
    } 
}
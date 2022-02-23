//date: 2022-02-23T16:51:14Z
//url: https://api.github.com/gists/5994671bd2cb869780343fad8bb41361
//owner: https://api.github.com/users/driedtoast

import java.util.concurrent.CompletionStage;
import javax.enterprise.context.ApplicationScoped;
import org.eclipse.microprofile.reactive.messaging.Incoming;
import org.eclipse.microprofile.reactive.messaging.Message;
import io.smallrye.common.annotation.Blocking;

@ApplicationScoped
public class TesterConsumer {

  
  @Incoming("trash")
  @Blocking
  public CompletionStage<Void> consume(Message<String> stuff) {
    System.out.println("trash: " + stuff.getPayload() );
    return stuff.ack();
    
  }
  
  @Incoming("trashalso")
  @Blocking
  public CompletionStage<Void> consumeAlso(Message<String> stuff) {
    System.out.println("Trash also: " + stuff.getPayload() );
    return stuff.ack();
    
  }  
}

//date: 2022-02-23T16:51:14Z
//url: https://api.github.com/gists/5994671bd2cb869780343fad8bb41361
//owner: https://api.github.com/users/driedtoast

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import javax.annotation.PostConstruct;
import javax.enterprise.context.ApplicationScoped;
import javax.enterprise.event.Observes;
import javax.inject.Inject;
import org.eclipse.microprofile.reactive.messaging.Channel;
import org.eclipse.microprofile.reactive.messaging.Emitter;
import org.eclipse.microprofile.reactive.messaging.Message;
import org.eclipse.microprofile.reactive.messaging.Outgoing;
import org.reactivestreams.Publisher;
import io.quarkus.runtime.StartupEvent;
import io.smallrye.reactive.messaging.providers.connectors.ExecutionHolder;
import io.vertx.mutiny.core.Vertx;
import io.vertx.mutiny.core.WorkerExecutor;

@ApplicationScoped
public class TesterPublisher {

  @Inject
  @Channel("trash")
  Emitter<String> emitter;
  
  @Inject ExecutionHolder executionHolder;
  
  private Vertx vertx;

  int count = 0;
  
  boolean started() {
    return emitter != null;
  }
  
  
  Message<String> createMessage(String payload) {
    return  Message.of(payload, null, () -> {
      System.out.println("acking message: " + payload);
      return  CompletableFuture.completedFuture(null);
    });
  }
  
  public void send(String payload) {
    emitter.send(createMessage(payload));
  }
  
  // @Outgoing("trash")
  public Message<String> getString() {
    count++;
    if(count % 10 == 0) {
      return null;
    }
    return createMessage("this sucks " + count);
  }
  
  long id = -1;
  
  public void cancel() {
    holder.cancelPeriodicalTask(id);
  }
  
  void runPeriodically(long millis, Runnable runnable) {
    Context context = vertx.getOrCreateContext();
    Handler<Long> action = id -> {
      
      if (Vertx.currentContext() == context && Context.isOnEventLoopThread()) {
          runnable.run();
        } else {
          context.runOnContext(x -> runnable.run());
        }
    }
    vertx.setTimer(millis, action);
  
  }
  
  
  public void onStart(@Observes  StartupEvent ev) {
  
    vertx = executionHolder.vertx();
    id = runPeriodically( TimeUnit.MILLISECONDS.toMillis(3), () -> {
      for(int i = 0; i < 10; i++) {
        emitter.send(createMessage("Test-" + i));
      }
    }); 
    
  }
}
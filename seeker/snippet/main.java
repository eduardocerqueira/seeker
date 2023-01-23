//date: 2023-01-23T16:40:24Z
//url: https://api.github.com/gists/74d44531dd7261b1431da07734e52447
//owner: https://api.github.com/users/EnriqueL8

public class OAuthTestVerticle extends AbstractVerticle {

    public static void main(final String[] args) {
        final Vertx vertx = Vertx.vertx();
        final OAuthTestVerticle verticle = new OAuthTestVerticle();
        vertx.deployVerticle(verticle)
                .onSuccess(v -> System.out.println("Verticle started " + v))
                .onFailure(err -> System.out.println("Verticle failed to start " + err.getMessage()));
    }

    @Override
    public void start(final Promise<Void> startPromise) {
        final Router mainRouter = Router.router(vertx);
        final ClassLoader classLoader = getClass().getClassLoader();
        // CookieSessionStore seems bugged, leads to cast exception `java.lang.ClassCastException: class java.util.LinkedHashMap cannot be cast to class io.vertx.ext.web.handler.impl.UserHolder`
        mainRouter.route().handler(SessionHandler.create(CookieSessionStore.create(vertx, "test")));
        
        OAuth2Config options = "**********"

        OAuth2Auth oauth2 = OAuth2Auth.create(vertx, options)
          
        // Line from other one here
        OAuth2AuthHandler oauth2Handler = OAuth2AuthHandler.create(this.vertx, oauth2, "http://localhost:8888/callback")
 
        // Set up the call back for the handler
        oauth2Handler.setupCallback(mainRouter.route("/callback"));

        // protect a sub path with the provided handler
        mainRouter.route("/protected/*").handler(oauth2Handler);
        
        mainRouter
              .route("/protected/somepage")
              .handler(ctx -> {
                  provider.authenticate(ctx.user().principal())
                          .onSuccess(user -> {
                                      final String role = "**********"
                                      if (!role.equals("admin")) {
                                          ctx.response()
                                                  .setStatusMessage("Not authorized to access this resource")
                                                  .setStatusCode(403);
                                          return;
                                      }
                                      System.out.println(new JsonObject(user.attributes().getString("accessToken")).getString("role"));
                                      ctx.response().end("Hello Admin! Welcome to the protected resource!\n" + ctx.user().attributes().encodePrettily() + "\n" + ctx.user().principal().encodePrettily());
                                  }
                          )
                          .onFailure(err -> {
                              // Failed!
                          });

              });

      
      // Run the vertx server
      vertx.createHttpServer().requestHandler(mainRouter).listen(8888).onSuccess(http -> {
          startPromise.complete();
          System.out.println("HTTP server started on port 8888");
      })
      .onFailure(startPromise::fail);
    }
}
rintln("HTTP server started on port 8888");
      })
      .onFailure(startPromise::fail);
    }
}

//date: 2022-01-18T17:13:16Z
//url: https://api.github.com/gists/d281589522402f41f15c8d072c9b14bd
//owner: https://api.github.com/users/engleangs

@SpringBootApplication
public class Application implements CommandLineRunner {
    private static final Logger LOGGER = LoggerFactory.getLogger(Application.class);
    @Autowired
    private SmsService smsService;

    @Override
    public void run(String... args) throws Exception {
        LOGGER.info("running");
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            smsService.stop();
        }));
        smsService.start();
        Thread.currentThread().join();

    }

    public static void main(String[] args)  {
        SpringApplication.run(Application.class, args);
    }
}
//date: 2023-02-10T16:45:42Z
//url: https://api.github.com/gists/d85f039ab228594bb3042e8552b06d5b
//owner: https://api.github.com/users/zzpzaf

@RestController
//@Controller
public class AuthController {

    private final Log logger = LogFactory.getLog(getClass());


    @PostMapping("/login")
	public ResponseEntity<String> login() {

        String msg = "Hello from Login!";
        logger.info(msg);
        
        try {
            return new ResponseEntity<>(msg, HttpStatus.OK);    
       } catch (Exception e) {
            return new ResponseEntity<>(null, HttpStatus.INTERNAL_SERVER_ERROR);
        }

	}

    @GetMapping("/")
    public ResponseEntity<String> myhome() {

        String msg = "Hello from home!";
        logger.info(msg);

        try {
            return new ResponseEntity<>(msg, HttpStatus.OK);    
        } catch (Exception e) {
            return new ResponseEntity<>(null, HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }


    @GetMapping("/error")
    public ResponseEntity<String> error() {
        
        String msg = "An Error occured!";
        logger.info(msg);
        return new ResponseEntity<>(msg, HttpStatus.INTERNAL_SERVER_ERROR);

    }
    
}
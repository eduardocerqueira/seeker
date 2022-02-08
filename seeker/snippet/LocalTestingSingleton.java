//date: 2022-02-08T16:49:35Z
//url: https://api.github.com/gists/b7497307d43590d308a725c662fadda9
//owner: https://api.github.com/users/mattonem

import com.browserstack.local.Local;
import java.util.Random;
import java.nio.charset.Charset;

public class LocalTestingSingleton {

    private static Local bsLocal;
    public static String localID;
    
    private LocalTestingSingleton(){}
    
    
    public static LocalTestingSingleton getInstance(){
        if(bsLocal == null){
            byte[] array = new byte[7];
            new Random().nextBytes(array);
            String localID = new String(array, Charset.forName("UTF-8"));
            bsLocal = new Local();
            HashMap<String, String> bsLocalArgs = new HashMap<String, String>();
            bsLocalArgs.put("key", "<browserstack-accesskey>");
            bsLocalArgs.put("localIdentifier", localID);
            # starts the Local instance with the required arguments
            bsLocal.start(bsLocalArgs);
        }
        return bsLocal;
    }
}
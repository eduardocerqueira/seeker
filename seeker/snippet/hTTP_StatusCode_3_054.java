//date: 2022-10-04T17:18:05Z
//url: https://api.github.com/gists/92b221b36d5066fbb4b6847b9bfa9b8c
//owner: https://api.github.com/users/maydin61

package TasksGeneral;

import java.util.Scanner;

public class hTTP_StatusCode_3_054 {
    public static void main(String[] args) {
        Scanner scan=new Scanner(System.in);
        System.out.println("Enter the status code");
        int statusCode= scan.nextInt();
        switch (statusCode){
            case 200:     // checking single case with multiple variable
                System.out.println("OK");
                break; // used to change and break out the follow
            case 201:
                System.out.println("Great");
                break;
            case 202:
                System.out.println("Accepted");
                break;
            case 301:
                System.out.println("Move permanently");
                break;
            case 303:
                System.out.println("See others");
                break;
            case 304:
                System.out.println("Not modified");
                break;
            case 307:
                System.out.println("Temporarily directed");
                break;
            case 400:
                System.out.println("Bad request");
                break;
            case 401:
                System.out.println("Unauthorized");
                break;
            case 403:
                System.out.println("Forbidden");
                break;
            case 404:
                System.out.println("Not found");
                break;
            case 510:
                System.out.println("Internet service error");
                break;
            case 503:
                System.out.println("Service unavailable");
                break;

            default:   // like else statement final part
                System.out.println("Invalid status code");
            break;    // optional to use break comment in at the end

        }
    }
}

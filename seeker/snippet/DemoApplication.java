//date: 2024-12-18T17:05:20Z
//url: https://api.github.com/gists/62129fe81f8db007463b670ba5d97a5e
//owner: https://api.github.com/users/suvincent

package com.example.demo;

import io.vavr.control.Either;
import java.util.List;

public class DemoApplication {

	public static void main(String[] args) {

        interface Mail {
            String getEmail();   
        }
    
        
        // Concrete implementation
        class Email implements Mail {
            private final String email;
        
            public Email(String email) {
                this.email = email;
            }
            @Override
            public String getEmail() {
                return email;
            }
        }

        // Concrete implementation
        class Email2 implements Mail {
            private final String email;
        
            public Email2(String email) {
                this.email = email;
            }
            @Override
            public String getEmail() {
                return "email2=>"+email;
            }
        }

        

        RuleSet<String, Email> lengthRule = RuleSet.create(
            input -> input.length() > 5 
                ? Either.right(input) 
                : Either.left(new Exception("Input too short")), 
            Email.class
        );

        RuleSet<String, Email> emailRule = new RuleSet<>(
            input -> input.contains("@") 
                ? Either.right(input) 
                : Either.left(List.of(new Exception("Invalid email format"))), 
            Email.class
        );

        RuleSet<String, Email> specialWordsRule = new RuleSet<>(
            input -> input.contains(".com") 
                ? Either.right(input) 
                : Either.left(List.of(new Exception("Invalid email format no .com"))), 
            Email.class
        );

        // Combine rules using or()
        RuleSet<String, Email> combinedRule = lengthRule.or(emailRule).or(specialWordsRule);        

        // Validate input
        Either<List<Exception>, Email> result1 = combinedRule.validate("short");
        if(result1.isLeft()){
            System.out.println(result1.getLeft().toString());
        }else{
            System.out.println("success: "+result1.get().getEmail());
        }
        Either<List<Exception>, Email> result2 = combinedRule.validate("valid@email.com");
        if(result2.isLeft()){
            System.out.println(result2.getLeft().toString());
        }else{
            System.out.println("success: "+result2.get().getEmail());
        }
	}

}

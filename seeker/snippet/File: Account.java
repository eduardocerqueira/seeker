//date: 2023-01-31T16:51:29Z
//url: https://api.github.com/gists/ee10c357b6e3eb60fa844a4a8608b9db
//owner: https://api.github.com/users/Vishuu005

//A Account class which is a fully encapsulated class.  
//It has a private data member and getter and setter methods.  
class Account {  
//private data members  
private long acc_no;  
private String name,email;  
private float amount;  
//public getter and setter methods  
public long getAcc_no() {  
    return acc_no;  
}  
public void setAcc_no(long acc_no) {  
    this.acc_no = acc_no;  
}  
public String getName() {  
    return name;  
}  
public void setName(String name) {  
    this.name = name;  
}  
public String getEmail() {  
    return email;  
}  
public void setEmail(String email) {  
    this.email = email;  
}  
public float getAmount() {  
    return amount;  
}  
public void setAmount(float amount) {  
    this.amount = amount;  
}  
  
} 
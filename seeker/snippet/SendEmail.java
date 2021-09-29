//date: 2021-09-29T16:51:38Z
//url: https://api.github.com/gists/d7aff6111488e0e956a6454b23e5d1f1
//owner: https://api.github.com/users/leroyvv

public class ProgramImpl
  extends com.tridium.program.ProgramBase
{

////////////////////////////////////////////////////////////////
// Getters
////////////////////////////////////////////////////////////////

  public BEmailAddress getFrom() { return (BEmailAddress)get("from"); }
  public BEmailAddressList getTo() { return (BEmailAddressList)get("to"); }

////////////////////////////////////////////////////////////////
// Setters
////////////////////////////////////////////////////////////////

  public void setFrom(javax.baja.email.BEmailAddress v) { set("from", v); }
  public void setTo(javax.baja.email.BEmailAddressList v) { set("to", v); }

////////////////////////////////////////////////////////////////
// Program Source
////////////////////////////////////////////////////////////////

  public void onStart() throws Exception
  {
    // start up code here
  }
  
  public void onExecute() throws Exception
  {
    // execute code (set executeOnChange flag on inputs)
  }
  
  public void onStop() throws Exception
  {
    // shutdown code here
  }                       
  
  public void onSend() throws Exception
  {
    // create email and set to/from
    BEmail email = new BEmail();
    email.setFrom(getFrom());
    email.setTo(getTo());          
    
    // create message
    email.setBody(new BTextPart("hello world\nline2!"));
  
    // add text attachment
    email.getAttachments().add("attach1", new BTextPart("file1.txt", "text 1 attachment\nline2!"));
    email.getAttachments().add("attach2", new BTextPart("file2.txt", "text 2 attachment\nline2!"));
    
      // lookup service and send
    BEmailService service = (BEmailService)Sys.getService(BEmailService.TYPE);
    service.send(email);
  }
  
}
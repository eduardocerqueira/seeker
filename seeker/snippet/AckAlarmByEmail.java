//date: 2021-09-29T16:51:38Z
//url: https://api.github.com/gists/d7aff6111488e0e956a6454b23e5d1f1
//owner: https://api.github.com/users/leroyvv

public class ProgramImpl
  extends com.tridium.program.ProgramBase
{

////////////////////////////////////////////////////////////////
// Program Source
////////////////////////////////////////////////////////////////

  public void onStart() throws Exception
  {
    if (log.isLoggable(Level.FINE)) log.fine("Starting AckAlarmViaEmailProgram");
  }   
  
  public void onExecute() throws Exception {}
  
  /**
   * Called when a mail is received
   *
   * How to use
   * ----------
   *
   * 1. Set up the EmailService with the OutgoingAccount and IncomingAccount components
   * 2. Use the EmailRecipient to retransmit the desired alarm information.
   * 3. In the 'Subject' Property of the EmailRecipient add 'Alarm -> UUID:%uuid%'. Please note, other text can go in here as well if desired!
   * 4. Connect this Program's receive Action to the 'received' Topic of the IncomingAccount
   * 5. To turn the tracing on for debugging, go to the Debug Service page and select 'FINE' for AckAlarmViaEmail!
   */
  public void onReceive(BEmail email) throws Exception
  {     
    //
    // Only ack the alarm if we receive an email from one of the users under the UserService
    // 
    
    String emailAddress = email.getFrom().getAddress();
    
    if (log.isLoggable(Level.FINE)) log.fine("Received email from " + emailAddress);
      
    StringBuffer buff = new StringBuffer();
    buff.append("service:baja:UserService|bql:select from baja:User where email = '");
    buff.append(emailAddress); 
    buff.append("'");
    
    
    BUser user = null;
    //resolve bql query, open cursor of results and close with try-with-resources
    try (Cursor c = ((BITable)BOrd.make(buff.toString()).get(getComponent())).cursor())
    {
      //get the first user with a matching email address
      if (c.next())
      {
        user = (BUser)c.get();
      }
    }
    
    if (user == null)
    {
      log.severe("Alarm ack aborted. Could not from unregistered user: " + emailAddress);
      return;
    } 
    
    //
    // Find the UUID (the unique identifier for each alarm) from the returned subject
    //
    
    int uuidIndex = email.getSubject().indexOf(UUID_TEXT);
    
    if (uuidIndex <= -1)
    {
      log.severe("Alarm ack aborted. Could not find UUID in subject"); 
      return;
    }  
    
    // Get the UUID from the emails subject header
    String uuidStr = email.getSubject().substring(uuidIndex + UUID_TEXT.length(), uuidIndex + 36 + UUID_TEXT.length()); 
    
    System.out.println("Got uuid: " + uuidStr);
    
    // Form alarm id
    BUuid alarmId = (BUuid)BUuid.DEFAULT.decodeFromString(uuidStr);
    
    //
    // Acknowledge the alarm
    //  
    
    // Get the alarm database
    BAlarmService alarmService = (BAlarmService)Sys.getService(BAlarmService.TYPE);
    
    // Look up the alarm
    BAlarmRecord record = null;
    try (AlarmDbConnection conn = alarmService.getAlarmDb().getDbConnection(null))
    {
      record = conn.getRecord(alarmId);
    
      if (record == null)
      {
        log.warning("Could not find alarm to ack: " + alarmId);
        return;
      }
      else if (!record.getAckRequired())
      {
        log.warning("Alarm does not need acknowleding: " + alarmId);
        return;
      }
      else if (record.isAcknowledged())
      {
        log.warning("Alarm already acknowledged by " + record.getUser());
        return;
      }  
    
      // Ack the alarm with the user name
      record.ackAlarm(user.getName());
    
      // Update the alarm database with the new record
      conn.update(record); 
    }
    
    // Ack the alarm in the alarm service
    alarmService.ackAlarm(record);
    
    if (log.isLoggable(Level.FINE)) log.fine("Acked alarm from " + emailAddress + " -> " + uuidStr);   
    
  }
  
  public void onStop() throws Exception {}     
  
  private static final String UUID_TEXT = "UUID:"; 
  private static final Logger log = Logger.getLogger("AckAlarmViaEmail");
  
}
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

  
  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Sample Code
  //////////////////////////////////////////////////////////////////////////////////////////////////
  
    /**
     * This action:
     *   - executes bql query which returns a ITable
     *   - formats the table as an in-memory CSV file
     *   - emails the CSV file as an attachment
     */
    public void onSend() throws Exception
    {                     
      OrdTarget table = query();
      String csv = exportToCsv(table);
      email("table.csv", csv);             
    }       
    
    /**
     * Perform a bql query which returns an OrdTarget for a BITable
     */
    private OrdTarget query()
      throws Exception
    { 
      BComponent base = getComponent();                           
      return BOrd.make("slot:/|bql:select toPathString, type, toString from baja:Component").resolve(base);
    }
    
    /**
     * Run the CSV exporter against the specified table to build an
     * in memory representation of the table as a CSV file.
     */
    private String exportToCsv(OrdTarget table) 
      throws Exception
    { 
      // create instance of ITableToCsv exporter                        
      BExporter exporter = (BExporter)Sys.getType("file:ITableToCsv").getInstance(); 
      
      // run the CSV exporter to export to memory based byte array
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      ExportOp op = ExportOp.make(table, out);  
      exporter.export(op);
      
      // return as string (this works because we String will use the default 
      // encoding, which should match encoding ITableToCsv exporter used to 
      // create a PrintWriter from a raw OutputStream)    
      return new String(out.toByteArray());
    }
    
    /**
     * Send the email with the specified attachment 
     */
    private void email(String fileName, String attachment) 
      throws Exception
    {
      // create email and set to/from
      BEmail email = new BEmail();
      email.setFrom(getFrom());
      email.setTo(getTo());          
      
      // create message        
      email.setSubject("Niagara BQL Results");
      email.setBody(new BTextPart("Niagara BQL Results in attachment!"));
    
      // add text attachment
      email.getAttachments().add("attach1", new BTextPart(fileName, attachment));
      
        // lookup service and send
      BEmailService service = (BEmailService)Sys.getService(BEmailService.TYPE);
      service.send(email);
    }
  
  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Lifecycle Callbacks
  //////////////////////////////////////////////////////////////////////////////////////////////////
  
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
  
}
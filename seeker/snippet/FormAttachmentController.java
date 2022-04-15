//date: 2022-04-15T16:51:14Z
//url: https://api.github.com/gists/f2168bd8affbbb45780890a10c1d3476
//owner: https://api.github.com/users/shruti910

@RestController
public class FormAttachmentController {	
	
private static final Logger logger = LoggerFactory.getLogger(FormAttachmentController.class);

	@Autowired private ClamAVServiceUtil clamavService;
	
	@RequestMapping(value = "/FileUpload",method =RequestMethod.POST,consumes = {MediaType.MULTIPART_FORM_DATA_VALUE},produces= {MediaType.APPLICATION_JSON_VALUE})
	public ResponseEntity<?> scanFile(
			@RequestParam(value="file", required = false) MultipartFile file,
			@RequestParam (value="attachmentName", required = false) String attachmentName,
			@RequestParam ("comments") String comments)
			throws MessagingException , Exception{
    
			logger.info("Scanning the file attachement..");
			
			try {
				Boolean isFilesafe = clamavService.scanFileAttachment(attachmentName, file);
				if(isFilesafe) {
					 //process further..
				}
				else {
			    		logger.info("Error: Infected file found!");
          				  // uploaded file is infected
				}
			}catch(Exception e) {
				logger.error(e.getMessage());
			}
	return null; //return response as per your logic.
	}

}
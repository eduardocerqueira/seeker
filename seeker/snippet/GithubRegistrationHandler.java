//date: 2022-01-25T16:54:56Z
//url: https://api.github.com/gists/669e4639387b5f97b83897c20946560e
//owner: https://api.github.com/users/ggit007

global with sharing class GithubRegistrationHandler extends Auth.AuthProviderPluginClass {

    private String customMetadataTypeApiName = 'Github_Provider__mdt';
    public  String redirectUrl; 
    private String key;
    private String secret;
    private String authUrl;  
    private String accessTokenUrl; 
    private String userInfoUrl; 
    private String scope;
    
    global String getCustomMetadataType(){
        return customMetadataTypeApiName;
    }
    
    /* Step 1 */
    global PageReference initiate(Map<string,string> authProviderConfiguration, String stateToPropagate) { 
    
        key         = authProviderConfiguration.get('Consumer_Key__c');
        authUrl     = authProviderConfiguration.get('Auth_URL__c');
        scope       = authProviderConfiguration.get('Scope__c');
        redirectUrl = authProviderConfiguration.get('Callback_URL__c');
        String urlToRedirect = authUrl+'?client_id='+key+'&redirect_uri='+redirectUrl+'&scope='+scope+
                               '&state='+stateToPropagate+'&allow_signup=false';
                               
        PageReference pageRef = new PageReference(urlToRedirect);                    
        return pageRef; 
    } 
    
    /* Step 2 */
    global Auth.AuthProviderTokenResponse handleCallback(Map<string,string> authProviderConfiguration, Auth.AuthProviderCallbackState state ) { 
        
        // This will contain an optional accessToken and refreshToken 
        key = authProviderConfiguration.get('Consumer_Key__c'); 
        secret = authProviderConfiguration.get('Consumer_Secret__c'); 
        accessTokenUrl = authProviderConfiguration.get('Token_URL__c'); 
        redirectUrl = authProviderConfiguration.get('Callback_URL__c');
        
        Map<String,String> queryParams = state.queryParameters; 
        
        String code = queryParams.get('code'); 
        String sfdcState = queryParams.get('state'); 
        
        
        HttpRequest req = new HttpRequest(); 
        String requestBody = 'client_id='+key+'&client_secret='+secret+'&code='+code
                             +'&redirect_uri='+redirectUrl+'&state='+sfdcState;
        req.setEndpoint(accessTokenUrl); 
        req.setHeader('Accept','application/json'); 
        req.setMethod('POST'); 
        req.setBody(requestBody);
        Http http = new Http(); 
        
        HTTPResponse res = http.send(req); 
        String responseBody = res.getBody(); 
        GitHubWrapper wrapper = (GitHubWrapper)System.JSON.deserialize(responseBody, GitHubWrapper.class);
        
        return new Auth.AuthProviderTokenResponse('GithubRegistrationHandler', wrapper.access_token, 'refreshToken', sfdcState); 
        
    } 
   
               
    global Auth.UserData getUserInfo(Map<string,string> authProviderConfiguration, Auth.AuthProviderTokenResponse response) { 
        userInfoUrl = authProviderConfiguration.get('User_Info_URL__c');
        
        HttpRequest req = new HttpRequest(); 
        
        req.setEndpoint(userInfoUrl); 
        req.setHeader('Content-Type','application/json'); 
        req.setMethod('GET'); 
        req.setHeader('Authorization', 'Bearer '+response.oauthToken);
        Http http = new Http(); 
        HTTPResponse res = http.send(req); 
        
        String responseBody = res.getBody();
        
        GithubUserWrapper userInfo = (GithubUserWrapper)System.JSON.deserialize(responseBody, GithubUserWrapper.class);
        
        List<String> nameInfo = userInfo.name.split(' ');
        
        Map<String,String> attributeMap = new Map<String,String>{'noauth' => 'NOUTHS'};
        
        Auth.UserData userdata = new Auth.UserData(userInfo.login, nameInfo.get(0), nameInfo.get(1), 
                 userInfo.name, userInfo.email, userInfo.url, userInfo.login, 'en_US', 'Github', null , attributeMap );
        
        System.debug('### userInfo '+userInfo);
        
        /*
            UserData(String identifier, String firstName, String lastName, String fullName, String email, 
                     String link, String userName, String locale, String provider, String siteLoginUrl, 
                     Map<String,String> attributeMap
            )
        */
        
        return userdata;
    } 
    
    
    public class GitHubWrapper {
        public String access_token; 
        public String scope;    
        public String token_type;
        
    }
    
    public class GithubUserWrapper{
        public String login;   
        public Integer id;  
        public String url;  
        public String html_url; 
        public String name; 
        public String company;  
        public String blog; 
        public String location;
        public String email;  
    }

}
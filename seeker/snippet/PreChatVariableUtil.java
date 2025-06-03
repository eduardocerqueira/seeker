//date: 2025-06-03T17:11:20Z
//url: https://api.github.com/gists/d92a85fd251b107f253a8b444ae2ad7a
//owner: https://api.github.com/users/jfwberg

public class PreChatVariableUtil {
    
    /**
     * @description Method get the shared secret
     * @note		!! Needs to come from a proper secret management solution !!
     */ 
    public static String getSharedSecret(){
        return 'vjIWNsL/bpiPDcRHfOMqvdaOeFeCISWSBMRDJo/GIwM=';
    }
    
   
    /**
     * @description Example method to validate a hash. It does not validate 
     *              input for nulls, or checks if keys exist etc. It's purely
     *              to outline how a method like this works.
     * @return      True if the hash matches, else false
     */
    public static Boolean validateHash(String userId, String contextData, 
                                String contextHash){
    
        // In this example we use the Salesforce Id as the Salt, but it can 
        // be a second shared secret as well, As long as both 
        // parties know it and can keep it secret
        String salt = [SELECT Id FROM Contact WHERE External_Id__c =:userId]?.Id;
        
        // Call the hash function and generate the hash to validate
        String hash = generateHash(contextData, salt);
    
        // return true if the hashes match
        return hash == contextHash;
    }

    
    /**
     * @description Method that generates a hash based on a value and a salt
     * @return      A base64 encoded SHA256 hash
     */
    public static String generateHash(String contextData, String salt) {
        
        // Data to hash: context data + a period + the salt
        Blob dataToHash = Blob.valueOf(contextData + '.' + salt);
        
        // Generate a hash from the data to hash
        Blob hashedData = Crypto.generateDigest('SHA256', dataToHash);
        
        // Return a base64 encoded hash
        return EncodingUtil.base64Encode(hashedData);
    }
    
    
    /**
     * @description Method to decrypt the data using the shared secret
     * @return      The decrypted input data
     */
    public static String decrypt(String encryptedData){
        return 
            Crypto.decryptWithManagedIV(
                'AES256',
                EncodingUtil.base64Decode(getSharedSecret()),
                EncodingUtil.base64Decode(encryptedData)
            ).toString();
        
    }
    
    /**
     * @description Method to enncrypt data using the shared secret
     */
    public static String encrypt(String plainTextData){
        return EncodingUtil.base64Encode(
            Crypto.encryptWithManagedIV(
                'AES256',
                EncodingUtil.base64Decode(getSharedSecret()),
                Blob.valueOf(plainTextData)
            )
        );
    }
}
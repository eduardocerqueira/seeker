//date: 2023-04-26T16:56:26Z
//url: https://api.github.com/gists/70374dbf0b94bb882770ad6ce56bda46
//owner: https://api.github.com/users/Vergil333

/**
 * @property createdById Lookup(User) , 
 * @property createdDate Date/Time , 
 * @property id Lookup() , 
 * @property isDeleted Checkbox , 
 * @property lastModifiedById Lookup(User) , 
 * @property lastModifiedDate Date/Time , 
 * @property name Text(80) , 
 * @property ownerId Lookup(User,Group) , 
 * @property systemModstamp Date/Time , 
 * @property userRecordAccessId Lookup(User Record Access) , 
 */
public class TestObject__c extends AbstractSObject {
    @JsonProperty("CreatedById") public String createdById;

    public String getCreatedById() {
        return createdById;
    }

    public void setCreatedById(String createdById) {
        this.createdById = createdById;
    }


    @JsonProperty("CreatedDate") public ZonedDateTime createdDate;

    public ZonedDateTime getCreatedDate() {
        return createdDate;
    }

    public void setCreatedDate(ZonedDateTime createdDate) {
        this.createdDate = createdDate;
    }


    @JsonProperty("IsDeleted") public Boolean isDeleted;

    public Boolean getIsDeleted() {
        return isDeleted;
    }

    public void setIsDeleted(Boolean isDeleted) {
        this.isDeleted = isDeleted;
    }


    @JsonProperty("LastModifiedById") public String lastModifiedById;

    public String getLastModifiedById() {
        return lastModifiedById;
    }

    public void setLastModifiedById(String lastModifiedById) {
        this.lastModifiedById = lastModifiedById;
    }


    @JsonProperty("LastModifiedDate") public ZonedDateTime lastModifiedDate;

    public ZonedDateTime getLastModifiedDate() {
        return lastModifiedDate;
    }

    public void setLastModifiedDate(ZonedDateTime lastModifiedDate) {
        this.lastModifiedDate = lastModifiedDate;
    }


    @JsonProperty("Name") public String name;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }


    @JsonProperty("OwnerId") public String ownerId;

    public String getOwnerId() {
        return ownerId;
    }

    public void setOwnerId(String ownerId) {
        this.ownerId = ownerId;
    }


    @JsonProperty("SystemModstamp") public ZonedDateTime systemModstamp;

    public ZonedDateTime getSystemModstamp() {
        return systemModstamp;
    }

    public void setSystemModstamp(ZonedDateTime systemModstamp) {
        this.systemModstamp = systemModstamp;
    }


    @JsonProperty("UserRecordAccessId") public String userRecordAccessId;

    public String getUserRecordAccessId() {
        return userRecordAccessId;
    }

    public void setUserRecordAccessId(String userRecordAccessId) {
        this.userRecordAccessId = userRecordAccessId;
    }

   public TestObject__c() {
        super(SObjectType.TEST_OBJECT__C);
    }
}
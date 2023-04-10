//date: 2023-04-10T16:51:46Z
//url: https://api.github.com/gists/731de61ff8e238264915d05e8cc58322
//owner: https://api.github.com/users/Vergil333

/**
 * @property accountNumber Text(40) , 
 * @property accountSource Picklist , 
 * @property active__c Picklist , 
 * @property annualRevenue Currency(18, 0) , 
 * @property billingAddress Address , 
 * @property cleanStatus Picklist , 
 * @property createdById Lookup(User) , 
 * @property createdDate Date/Time , 
 * @property customerPriority__c Picklist , 
 * @property dandbCompanyId Lookup(D&B Company) , 
 * @property description Long Text Area(32000) , 
 * @property dunsNumber Text(9) , 
 * @property fax Fax , 
 * @property id Lookup() , 
 * @property industry Picklist , 
 * @property isDeleted Checkbox , 
 * @property jigsaw Text(20) , 
 * @property jigsawCompanyId External Lookup , 
 * @property lastActivityDate Date , 
 * @property lastModifiedById Lookup(User) , 
 * @property lastModifiedDate Date/Time , 
 * @property lastReferencedDate Date/Time , 
 * @property lastViewedDate Date/Time , 
 * @property masterRecordId Lookup(Account) , 
 * @property naicsCode Text(8) , 
 * @property naicsDesc Text(120) , 
 * @property name Name , 
 * @property numberOfEmployees Number(8, 0) , 
 * @property numberofLocations__c Number(3, 0) , 
 * @property operatingHoursId Lookup(Operating Hours) , 
 * @property ownerId Lookup(User) , 
 * @property ownership Picklist , 
 * @property parentId Hierarchy , 
 * @property phone Phone , 
 * @property photoUrl URL(255) , 
 * @property rating Picklist , 
 * @property sLAExpirationDate__c Date , 
 * @property sLASerialNumber__c Text(10) , 
 * @property sLA__c Picklist , 
 * @property shippingAddress Address , 
 * @property sic Text(20) , 
 * @property sicDesc Text(80) , 
 * @property site Text(80) , 
 * @property systemModstamp Date/Time , 
 * @property tickerSymbol Content(20) , 
 * @property tradestyle Text(255) , 
 * @property type Picklist , 
 * @property upsellOpportunity__c Picklist , 
 * @property userRecordAccessId Lookup(User Record Access) , 
 * @property website URL(255) , 
 * @property yearStarted Text(4) , 
 */
public class Account extends AbstractSObject {
    @JsonProperty("AccountNumber") public String accountNumber;

    public String getAccountNumber() {
        return accountNumber;
    }

    public void setAccountNumber(String accountNumber) {
        this.accountNumber = accountNumber;
    }


    @JsonProperty("AccountSource") public String accountSource;

    public String getAccountSource() {
        return accountSource;
    }

    public void setAccountSource(String accountSource) {
        this.accountSource = accountSource;
    }


    @JsonProperty("Active__c") public String active__c;

    public String getActive__c() {
        return active__c;
    }

    public void setActive__c(String active__c) {
        this.active__c = active__c;
    }


    @JsonProperty("AnnualRevenue") public Double annualRevenue;

    public Double getAnnualRevenue() {
        return annualRevenue;
    }

    public void setAnnualRevenue(Double annualRevenue) {
        this.annualRevenue = annualRevenue;
    }


    @JsonProperty("BillingAddress") public Map<String, Any?> billingAddress;

    public Map<String, Any?> getBillingAddress() {
        return billingAddress;
    }

    public void setBillingAddress(Map<String, Any?> billingAddress) {
        this.billingAddress = billingAddress;
    }


    @JsonProperty("CleanStatus") public String cleanStatus;

    public String getCleanStatus() {
        return cleanStatus;
    }

    public void setCleanStatus(String cleanStatus) {
        this.cleanStatus = cleanStatus;
    }


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


    @JsonProperty("CustomerPriority__c") public String customerPriority__c;

    public String getCustomerPriority__c() {
        return customerPriority__c;
    }

    public void setCustomerPriority__c(String customerPriority__c) {
        this.customerPriority__c = customerPriority__c;
    }


    @JsonProperty("DandbCompanyId") public String dandbCompanyId;

    public String getDandbCompanyId() {
        return dandbCompanyId;
    }

    public void setDandbCompanyId(String dandbCompanyId) {
        this.dandbCompanyId = dandbCompanyId;
    }


    @JsonProperty("Description") public String description;

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }


    @JsonProperty("DunsNumber") public String dunsNumber;

    public String getDunsNumber() {
        return dunsNumber;
    }

    public void setDunsNumber(String dunsNumber) {
        this.dunsNumber = dunsNumber;
    }


    @JsonProperty("Fax") public String fax;

    public String getFax() {
        return fax;
    }

    public void setFax(String fax) {
        this.fax = fax;
    }


    @JsonProperty("Industry") public String industry;

    public String getIndustry() {
        return industry;
    }

    public void setIndustry(String industry) {
        this.industry = industry;
    }


    @JsonProperty("IsDeleted") public Boolean isDeleted;

    public Boolean getIsDeleted() {
        return isDeleted;
    }

    public void setIsDeleted(Boolean isDeleted) {
        this.isDeleted = isDeleted;
    }


    @JsonProperty("Jigsaw") public String jigsaw;

    public String getJigsaw() {
        return jigsaw;
    }

    public void setJigsaw(String jigsaw) {
        this.jigsaw = jigsaw;
    }


    @JsonProperty("JigsawCompanyId") public String jigsawCompanyId;

    public String getJigsawCompanyId() {
        return jigsawCompanyId;
    }

    public void setJigsawCompanyId(String jigsawCompanyId) {
        this.jigsawCompanyId = jigsawCompanyId;
    }


    @JsonProperty("LastActivityDate") public LocalDate lastActivityDate;

    public LocalDate getLastActivityDate() {
        return lastActivityDate;
    }

    public void setLastActivityDate(LocalDate lastActivityDate) {
        this.lastActivityDate = lastActivityDate;
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


    @JsonProperty("LastReferencedDate") public ZonedDateTime lastReferencedDate;

    public ZonedDateTime getLastReferencedDate() {
        return lastReferencedDate;
    }

    public void setLastReferencedDate(ZonedDateTime lastReferencedDate) {
        this.lastReferencedDate = lastReferencedDate;
    }


    @JsonProperty("LastViewedDate") public ZonedDateTime lastViewedDate;

    public ZonedDateTime getLastViewedDate() {
        return lastViewedDate;
    }

    public void setLastViewedDate(ZonedDateTime lastViewedDate) {
        this.lastViewedDate = lastViewedDate;
    }


    @JsonProperty("MasterRecordId") public String masterRecordId;

    public String getMasterRecordId() {
        return masterRecordId;
    }

    public void setMasterRecordId(String masterRecordId) {
        this.masterRecordId = masterRecordId;
    }


    @JsonProperty("NaicsCode") public String naicsCode;

    public String getNaicsCode() {
        return naicsCode;
    }

    public void setNaicsCode(String naicsCode) {
        this.naicsCode = naicsCode;
    }


    @JsonProperty("NaicsDesc") public String naicsDesc;

    public String getNaicsDesc() {
        return naicsDesc;
    }

    public void setNaicsDesc(String naicsDesc) {
        this.naicsDesc = naicsDesc;
    }


    @JsonProperty("Name") public String name;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }


    @JsonProperty("NumberOfEmployees") public Double numberOfEmployees;

    public Double getNumberOfEmployees() {
        return numberOfEmployees;
    }

    public void setNumberOfEmployees(Double numberOfEmployees) {
        this.numberOfEmployees = numberOfEmployees;
    }


    @JsonProperty("NumberofLocations__c") public Double numberofLocations__c;

    public Double getNumberofLocations__c() {
        return numberofLocations__c;
    }

    public void setNumberofLocations__c(Double numberofLocations__c) {
        this.numberofLocations__c = numberofLocations__c;
    }


    @JsonProperty("OperatingHoursId") public String operatingHoursId;

    public String getOperatingHoursId() {
        return operatingHoursId;
    }

    public void setOperatingHoursId(String operatingHoursId) {
        this.operatingHoursId = operatingHoursId;
    }


    @JsonProperty("OwnerId") public String ownerId;

    public String getOwnerId() {
        return ownerId;
    }

    public void setOwnerId(String ownerId) {
        this.ownerId = ownerId;
    }


    @JsonProperty("Ownership") public String ownership;

    public String getOwnership() {
        return ownership;
    }

    public void setOwnership(String ownership) {
        this.ownership = ownership;
    }


    @JsonProperty("ParentId") public Map<String, Any?> parentId;

    public Map<String, Any?> getParentId() {
        return parentId;
    }

    public void setParentId(Map<String, Any?> parentId) {
        this.parentId = parentId;
    }


    @JsonProperty("Phone") public String phone;

    public String getPhone() {
        return phone;
    }

    public void setPhone(String phone) {
        this.phone = phone;
    }


    @JsonProperty("PhotoUrl") public String photoUrl;

    public String getPhotoUrl() {
        return photoUrl;
    }

    public void setPhotoUrl(String photoUrl) {
        this.photoUrl = photoUrl;
    }


    @JsonProperty("Rating") public String rating;

    public String getRating() {
        return rating;
    }

    public void setRating(String rating) {
        this.rating = rating;
    }


    @JsonProperty("SLAExpirationDate__c") public LocalDate sLAExpirationDate__c;

    public LocalDate getSLAExpirationDate__c() {
        return sLAExpirationDate__c;
    }

    public void setSLAExpirationDate__c(LocalDate sLAExpirationDate__c) {
        this.sLAExpirationDate__c = sLAExpirationDate__c;
    }


    @JsonProperty("SLASerialNumber__c") public String sLASerialNumber__c;

    public String getSLASerialNumber__c() {
        return sLASerialNumber__c;
    }

    public void setSLASerialNumber__c(String sLASerialNumber__c) {
        this.sLASerialNumber__c = sLASerialNumber__c;
    }


    @JsonProperty("SLA__c") public String sLA__c;

    public String getSLA__c() {
        return sLA__c;
    }

    public void setSLA__c(String sLA__c) {
        this.sLA__c = sLA__c;
    }


    @JsonProperty("ShippingAddress") public Map<String, Any?> shippingAddress;

    public Map<String, Any?> getShippingAddress() {
        return shippingAddress;
    }

    public void setShippingAddress(Map<String, Any?> shippingAddress) {
        this.shippingAddress = shippingAddress;
    }


    @JsonProperty("Sic") public String sic;

    public String getSic() {
        return sic;
    }

    public void setSic(String sic) {
        this.sic = sic;
    }


    @JsonProperty("SicDesc") public String sicDesc;

    public String getSicDesc() {
        return sicDesc;
    }

    public void setSicDesc(String sicDesc) {
        this.sicDesc = sicDesc;
    }


    @JsonProperty("Site") public String site;

    public String getSite() {
        return site;
    }

    public void setSite(String site) {
        this.site = site;
    }


    @JsonProperty("SystemModstamp") public ZonedDateTime systemModstamp;

    public ZonedDateTime getSystemModstamp() {
        return systemModstamp;
    }

    public void setSystemModstamp(ZonedDateTime systemModstamp) {
        this.systemModstamp = systemModstamp;
    }


    @JsonProperty("TickerSymbol") public String tickerSymbol;

    public String getTickerSymbol() {
        return tickerSymbol;
    }

    public void setTickerSymbol(String tickerSymbol) {
        this.tickerSymbol = tickerSymbol;
    }


    @JsonProperty("Tradestyle") public String tradestyle;

    public String getTradestyle() {
        return tradestyle;
    }

    public void setTradestyle(String tradestyle) {
        this.tradestyle = tradestyle;
    }


    @JsonProperty("Type") public String type;

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }


    @JsonProperty("UpsellOpportunity__c") public String upsellOpportunity__c;

    public String getUpsellOpportunity__c() {
        return upsellOpportunity__c;
    }

    public void setUpsellOpportunity__c(String upsellOpportunity__c) {
        this.upsellOpportunity__c = upsellOpportunity__c;
    }


    @JsonProperty("UserRecordAccessId") public String userRecordAccessId;

    public String getUserRecordAccessId() {
        return userRecordAccessId;
    }

    public void setUserRecordAccessId(String userRecordAccessId) {
        this.userRecordAccessId = userRecordAccessId;
    }


    @JsonProperty("Website") public String website;

    public String getWebsite() {
        return website;
    }

    public void setWebsite(String website) {
        this.website = website;
    }


    @JsonProperty("YearStarted") public String yearStarted;

    public String getYearStarted() {
        return yearStarted;
    }

    public void setYearStarted(String yearStarted) {
        this.yearStarted = yearStarted;
    }

   public Account() {
        super(SObjectType.ACCOUNT);
    }
}
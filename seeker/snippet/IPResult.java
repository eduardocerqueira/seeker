//date: 2022-07-26T16:45:09Z
//url: https://api.github.com/gists/0a3ef41bbf0084c257cc3f799b7359a0
//owner: https://api.github.com/users/CB-javiervilarsanchez

public class IPResult {
    static final String NOT_SUPPORTED = "Not_Supported";
    String ip_address;
    String country_short;
    String country_long;
    String region;
    String city;
    String isp;
    float latitude;
    float longitude;
    String domain;
    String zipcode;
    String netspeed;
    String timezone;
    String iddcode;
    String areacode;
    String weatherstationcode;
    String weatherstationname;
    String mcc;
    String mnc;
    String mobilebrand;
    float elevation;
    String usagetype;
    String addresstype;
    String category;
    String status;
    boolean delay = false;
    String version = "Version 8.9.1";

    IPResult(String ipstring) {
        ip_address = ipstring;
    }
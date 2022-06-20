//date: 2022-06-20T16:53:13Z
//url: https://api.github.com/gists/2ed09268303f0cdbc4619f1fe2db250e
//owner: https://api.github.com/users/TonyLuo

    // Used in various role mappers
    public static RoleModel getRoleFromString(RealmModel realm, String roleName) {
        String[] parsedRole = parseRole(roleName);
        RoleModel role = null;
        if (parsedRole[0] == null) {
            role = realm.getRole(parsedRole[1]);
        } else {
            ClientModel client = realm.getClientByClientId(parsedRole[0]);
            if (client != null) {
                role = client.getRole(parsedRole[1]);
            }
        }
        return role;
    }
    
     // Used for hardcoded role mappers
    public static String[] parseRole(String role) {
        int scopeIndex = role.lastIndexOf('.');
        if (scopeIndex > -1) {
            String appName = role.substring(0, scopeIndex);
            role = role.substring(scopeIndex + 1);
            String[] rtn = {appName, role};
            return rtn;
        } else {
            String[] rtn = {null, role};
            return rtn;

        }
    }
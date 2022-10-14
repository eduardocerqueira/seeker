//date: 2022-10-14T17:23:29Z
//url: https://api.github.com/gists/5823c706104acfd86bcb2a6db7c0084f
//owner: https://api.github.com/users/kivan-mih

String! constructString(Data! mainData, Data additionalData){
     return mainData.toString() 
           + additionalData?.map(ifNoNull -> "_" + ifNonNull.toString()).orElse("");
}
//date: 2021-12-16T17:11:37Z
//url: https://api.github.com/gists/ce1dcee81f64cdc8b2a8b9e9dc9446ca
//owner: https://api.github.com/users/the-dvlpr

List<Object> fieldList = (List<Object>)JSON.deserializeUntyped('[{"field":"phone","object":"account"},{"field":"name","object":"account"}]');
for(Object fld : fieldList){    
    Map<String,Object> data = (Map<String,Object>)fld;
    
    //Magic!
    system.debug(data.get('field'));
}
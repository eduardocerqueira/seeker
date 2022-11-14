//date: 2022-11-14T17:10:08Z
//url: https://api.github.com/gists/2192c81ae0451efe26e9501bae3e9921
//owner: https://api.github.com/users/aleksandr-dudko

// send the /predict request.
CompletableFuture<JsonNode> future = mlemClient.predict(requestBody);
// get the response.
JsonNode response1 = future.get();
// to handle an exception use exceptionally method.
JsonNode response2 = future
    .exceptionally(throwable -> {
        //handle exception here
        return null;
    }).get();
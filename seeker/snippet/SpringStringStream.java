//date: 2022-02-23T16:50:00Z
//url: https://api.github.com/gists/49b2aa5cbcccde0f1a1c1ff2ef7ab9eb
//owner: https://api.github.com/users/julianoBRL

@GetMapping
@SuppressWarnings({ "deprecation" })
private ResponseEntity<StreamingResponseBody> get() throws IOException {

  StreamingResponseBody responseBody = response -> {
       response.write(("{\"Header\":\"data\",").getBytes());
       otherClass(response);
       response.write(("}").getBytes());

       response.flush();
    };

    return ResponseEntity.ok()
          .contentType(MediaType.APPLICATION_STREAM_JSON)
          .body(responseBody);
}
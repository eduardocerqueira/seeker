//date: 2023-11-24T16:52:59Z
//url: https://api.github.com/gists/3436fd1f2aa6e4c78ca7bf9dc8392e60
//owner: https://api.github.com/users/nitinkc

@Test
void getUuidId() {
    try(MockedStatic<UuidUtils> utilities = Mockito.mockStatic(UuidUtils.class)){
        UUID uuid = UUID.randomUUID();
        Mockito.when(UuidUtils.getUuidId("12345")).thenReturn(uuid.toString());
        Assertions.assertEquals(UuidUtils.getUuidId("12345"),uuid.toString());
    }
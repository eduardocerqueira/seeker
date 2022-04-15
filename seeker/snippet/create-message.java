//date: 2022-04-15T17:11:51Z
//url: https://api.github.com/gists/15c98b166292e7112871f6370c036733
//owner: https://api.github.com/users/aspose-com-gists

// Get graph client
IGraphClient client = GraphClient.getClient(tokenProvider);

// Create message object and set properties
MapiMessage message = new MapiMessage();
message.setSubject("Subject");
message.setBody("Body");
message.setProperty(KnownPropertyList.DISPLAY_TO, "to@host.com");
message.setProperty(KnownPropertyList.SENDER_NAME, "from");
message.setProperty(KnownPropertyList.SENT_REPRESENTING_EMAIL_ADDRESS, "from@host.com");

// Create message in inbox
MapiMessage createdMessage = client.createMessage(GraphKnownFolders.Inbox, message);
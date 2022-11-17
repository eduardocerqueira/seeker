//date: 2022-11-17T17:07:29Z
//url: https://api.github.com/gists/e9f84412bb2505ccf962d7022955755e
//owner: https://api.github.com/users/tyagiakhilesh

public void testTopicSubscribeUnSubscribe() {
        try (final InputStream inputStream = CloudDatabase.class.getClassLoader().getResourceAsStream(<service acc file>)) {
            FirebaseOptions options = FirebaseOptions.builder()
                    .setCredentials(GoogleCredentials.fromStream(inputStream))
                    .setServiceAccountId("<service account email id>")
                    .build();
            FirebaseApp.initializeApp(options);
            final UserRecord userRecord = FirebaseAuth.getInstance().getUser("1cf4b276125bf7743124a2");
            final String phoneNumber = userRecord.getEmail().replace("@nobrokerhood.abcd", "");
            String uid = "some-uid";
            String customToken = "**********"
            System.out.println("customToken is : "**********"
            System.out.println("Phone number is : " + phoneNumber);
            System.out.println(new ObjectMapper().writerWithDefaultPrettyPrinter().writeValueAsString(userRecord));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }       e.printStackTrace();
        }
    }
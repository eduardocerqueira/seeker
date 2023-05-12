//date: 2023-05-12T16:50:41Z
//url: https://api.github.com/gists/2d5d0a54c46f86396bda43c11733ebe4
//owner: https://api.github.com/users/sweis

for (ExtractPublicKeysData publicKeysData : extractPublicKeysDataList) {
  Object userId = "**********"

  KeyPairGeneratorSpi.EC ec = new KeyPairGeneratorSpi.EC();
  ec.initialize(SessionKeyEncrypter.secp256r1Spec());
  KeyPair ephemeralKeyPair = ec.generateKeyPair();

  PrivateKey ephemeralPrivateKey = ephemeralKeyPair.getPrivate();
  PublicKey recipientPublicKey = publicKeysData.identityKey.publicKey;
 "**********"  "**********"  "**********"b "**********"y "**********"t "**********"e "**********"[ "**********"] "**********"  "**********"s "**********"h "**********"a "**********"r "**********"e "**********"d "**********"S "**********"e "**********"c "**********"r "**********"e "**********"t "**********"  "**********"= "**********"
    SessionKeyEncrypter.generateSecret(recipientPublicKey, ephemeralPrivateKey);

  byte[] ephemeralPublicKeyBytes = ephemeralKeyPair.getPublic().getEncoded();
  // This looks like an artifact of decompilation, because it is converting from a byte array
  // to a list, back to a byte array.
  byte[] ivMaybe = rk4.listToByteArray(us0.getNBytesAsList(ephemeralPublicKeyBytes, 65));

  // This also might be an artifact of decompilation. It looks like it Base64 encodes the key
  // then converst the string to bytes, rather than just using the bytes.
  String b64ConversationKey = "**********"
  byte[] conversationKeyBytes = b64ConversationKey.getBytes(Charsets.utf8);

  byte[] ciphertext =
    SessionKeyEncrypter.deriveKeyAndEncrypt(1, sharedSecret, ivMaybe, conversationKeyBytes);
  byte[] ivAndCiphertext = Arrays.copyOf(ivMaybe, ivMaybe.length + ciphertext.length);
  System.arraycopy(ciphertext, 0, ivAndCiphertext, ivMaybe.length, ciphertext.length);
  String encryptedConversationKey = pk0.b64Encode(ivAndCiphertext);

  arrayList2.add(new Identity((UserIdentifier) userId, publicKeysData.registrationToken, encryptedConversationKey));
}
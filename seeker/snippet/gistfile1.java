//date: 2023-06-06T17:06:17Z
//url: https://api.github.com/gists/ed50d5d8029c9c64ccc8246f2dc3e3d7
//owner: https://api.github.com/users/danielFesenmeyer

var filePath = Path.of("test.txt");

Files.writeString(filePath, "My test content");

var fileContent = Files.readString(filePath);
System.out.println(fileContent);

Files.delete(filePath);
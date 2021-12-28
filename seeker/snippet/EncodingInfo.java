//date: 2021-12-28T17:07:17Z
//url: https://api.github.com/gists/f23546ecd756a62a12ff21a7a84eeef5
//owner: https://api.github.com/users/TheDarkestDay

 public static void printEncodingInfo() {
        // Creating an array of byte type chars and
        // passing random  alphabet as an argument.abstract
        // Say alphabet be 'w'
        byte[] byte_array = { 'w' };

        // Creating an object of InputStream
        InputStream instream
                = new ByteArrayInputStream(byte_array);

        // Now, opening new file input stream reader
        InputStreamReader streamreader
                = new InputStreamReader(instream);
        String streamEncoding = streamreader.getEncoding();

        // Method returns a string of character encoding
        // used by using System.getProperty()
        String defaultEncoding
                = System.getProperty("file.encoding");

        System.out.println("Default Charset: "
                + defaultEncoding);

        // Getting character encoding by InputStreamReader
        System.out.println(
                "Default Charset by InputStreamReader: "
                        + streamEncoding);

        // Getting character encoding by java.nio.charset
        System.out.println("Default Charset: "
                + Charset.defaultCharset());
    }
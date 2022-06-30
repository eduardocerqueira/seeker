//date: 2022-06-30T17:19:26Z
//url: https://api.github.com/gists/b43b121fdbf255723aa8995181950ff5
//owner: https://api.github.com/users/CodeLover254

import org.apache.commons.codec.DecoderException;
import org.apache.commons.validator.routines.checkdigit.CheckDigitException;
import org.apache.commons.validator.routines.checkdigit.LuhnCheckDigit;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.util.Base64;

/**
 * Decoder Key Generation Algorithm 4 class
 */
public class Dkga04 {
    private final int KEYSIZE = 128;
    private final String DKGA = "04";
    private final String EA = "11";
    private final String BDT = "93";
    private final String IIN = "600727";
    private String keyType;
    private String supplyGroupCode;
    private String tariffIndex;
    private String keyRevisionNumber;
    private String decoderReferenceNumber;

    /**
     * @param keyType
     * @param supplyGroupCode
     * @param tariffIndex
     * @param keyRevisionNumber
     * @param decoderReferenceNumber
     */
    public Dkga04(String keyType, String supplyGroupCode, String tariffIndex, String keyRevisionNumber, String decoderReferenceNumber) {
        this.keyType = keyType;
        this.supplyGroupCode = supplyGroupCode;
        this.tariffIndex = tariffIndex;
        this.keyRevisionNumber = keyRevisionNumber;
        this.decoderReferenceNumber = decoderReferenceNumber;
    }

    /**
     * Gets the 160-bit key from a file
     * @return byte[]
     * @throws IOException
     */
    private byte[] getVendingKey() throws IOException{
        BufferedReader reader = new BufferedReader(new FileReader("160BitVendingKey.key"));
        byte[] key = Base64.getDecoder().decode(reader.readLine().trim());
        reader.close();
        return key;
    }

    /**
     * calculate the check digit given part of the PAN block
     * @param block
     * @return String
     * @throws CheckDigitException
     */
    private String calculateCheckDigit(String block) throws CheckDigitException {
        LuhnCheckDigit luhnCheckDigit = new LuhnCheckDigit();
        return luhnCheckDigit.calculate(block);
    }

    /**
     * Builds the 49-byte data block as a hex string
     * @return String
     * @throws CheckDigitException
     */
    private String constructDataBlock() throws CheckDigitException {
        String meterPan = IIN+decoderReferenceNumber;
        meterPan+=calculateCheckDigit(meterPan);

        StringBuilder builder = new StringBuilder();
        builder.append("0402");//initial separator
        builder.append(BinaryUtils.asciiToHex(DKGA));//DKGA
        builder.append("02");//separator
        builder.append(BinaryUtils.asciiToHex(EA));//EA
        builder.append("02");//separator
        builder.append(BinaryUtils.asciiToHex(BDT));//Base Date
        builder.append("02");//separator
        builder.append(BinaryUtils.asciiToHex(tariffIndex));//Tariff Index
        builder.append("000406");//mid separator
        builder.append(BinaryUtils.asciiToHex(supplyGroupCode));//Supply Group Code
        builder.append("01");//separator
        builder.append(BinaryUtils.asciiToHex(keyType));//Key Type
        builder.append("01");//separator
        builder.append(BinaryUtils.asciiToHex(keyRevisionNumber));//Key Revision Number
        builder.append("01");//separator
        builder.append(BinaryUtils.asciiToHex(meterPan));//Meter PAN
        builder.append(BinaryUtils.getPaddedString(BinaryUtils.decToHex(KEYSIZE),8));//Length of  Decoder Key

        return builder.toString();
    }

    /**
     * Computes the hmac sha256 hash using the vending key as the key.
     * @return String
     * @throws IOException
     * @throws NoSuchAlgorithmException
     * @throws InvalidKeyException
     * @throws DecoderException
     * @throws CheckDigitException
     */
    private String hmacSha256() throws IOException, NoSuchAlgorithmException, InvalidKeyException, DecoderException, CheckDigitException {
        byte[] key= getVendingKey();
        String dataBlock = constructDataBlock();
        Mac hmac = Mac.getInstance("HmacSHA256");
        SecretKeySpec keySpec = new SecretKeySpec(key,"HmacSHA256");
        hmac.init(keySpec);
        byte[] bytes = BinaryUtils.hexToByteArray(dataBlock);
        return BinaryUtils.byteArrayToHex(hmac.doFinal(bytes));
    }

    /**
     * Obtains and truncates the binary output
     * to remain with {KeySize} most significant bits
     * @return String
     * @throws DecoderException
     * @throws IOException
     * @throws NoSuchAlgorithmException
     * @throws InvalidKeyException
     * @throws CheckDigitException
     */
    public String generateDecoderKey() throws DecoderException, IOException, NoSuchAlgorithmException, InvalidKeyException, CheckDigitException {
        String hmacHash = hmacSha256();
        String binary = BinaryUtils.hexToBin(hmacHash,KEYSIZE);
        return BinaryUtils.binToHex(binary.substring(0,KEYSIZE));
    }
}

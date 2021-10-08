//date: 2021-10-08T17:16:32Z
//url: https://api.github.com/gists/7520e1dea6e79888acab8ea8206afe92
//owner: https://api.github.com/users/DinoChiesa

import org.bouncycastle.asn1.ASN1EncodableVector;
import org.bouncycastle.asn1.ASN1Integer;
import org.bouncycastle.asn1.ASN1Sequence;
import org.bouncycastle.asn1.DERSequence;
import java.math.BigInteger;

  private static byte[] toP1363(byte[] asn1EncodedSignature) throws Exception {
    ASN1Sequence seq = ASN1Sequence.getInstance(asn1EncodedSignature);
    BigInteger r = ((ASN1Integer) seq.getObjectAt(0)).getValue();
    BigInteger s = ((ASN1Integer) seq.getObjectAt(1)).getValue();
    int n = (r.bitLength() + 7) / 8;
    // round up to nearest even integer
    n = (int) Math.round((n+1)/2) * 2;
    byte[] out = new byte[2 * n];
    toFixed(r, out, 0, n);
    toFixed(s, out, n, n);
    return out;
  }

  private static byte[] toASN1(byte[] p1363EncodedSignature) throws IOException {
    int n = p1363EncodedSignature.length / 2;
    BigInteger r = new BigInteger(+1, Arrays.copyOfRange(p1363EncodedSignature, 0, n));
    BigInteger s = new BigInteger(+1, Arrays.copyOfRange(p1363EncodedSignature, n, n * 2));
    ASN1EncodableVector v = new ASN1EncodableVector();
    v.add(new ASN1Integer(r));
    v.add(new ASN1Integer(s));
    return new DERSequence(v).getEncoded();
  }

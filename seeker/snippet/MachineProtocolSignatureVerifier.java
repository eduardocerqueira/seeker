//date: 2021-11-02T17:03:56Z
//url: https://api.github.com/gists/2764b1664d480597d5e7152e9312c01c
//owner: https://api.github.com/users/Glamdring

public class MachineProtocolSignatureVerifier {
    public static void main(String[] args) throws Exception {
        String toVerify = FileUtils.readFileToString(new File("C:\\Users\\user\\Downloads\\010300076\\010300076.csv"));
        String signed = "<<base64 from p7s file>>";
        byte[] signedByte = Base64.getDecoder().decode(signed);

        Security.addProvider(new BouncyCastleProvider());

        CMSSignedData cms = new CMSSignedData(new CMSProcessableByteArray(toVerify.getBytes()), signedByte);
        
        Store<X509CertificateHolder> store = cms.getCertificates(); 
        SignerInformationStore signers = cms.getSignerInfos(); 
        for (SignerInformation signer : signers.getSigners()) { 
            Collection<X509CertificateHolder> certCollection = store.getMatches(signer.getSID()); 
            Iterator<X509CertificateHolder> certIt = certCollection.iterator();
            X509CertificateHolder certHolder = (X509CertificateHolder) certIt.next();
            X509Certificate cert = new JcaX509CertificateConverter().setProvider("BC").getCertificate(certHolder);
            if (signer.verify(new JcaSimpleSignerInfoVerifierBuilder().setProvider("BC").build(cert))) {
                System.out.println("verified");
            }
        }
    }
}
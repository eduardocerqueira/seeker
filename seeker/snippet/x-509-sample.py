#date: 2023-09-01T16:53:02Z
#url: https://api.github.com/gists/828f40f4a6c73da5301797dc966f6e63
#owner: https://api.github.com/users/Talismanic

#generating certificate and private key

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
import datetime

# Generate Private Key
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
    backend=default_backend()
)

# Generate Self-Signed Certificate
subject = issuer = x509.Name([
    x509.NameAttribute(NameOID.COUNTRY_NAME, "BN"),
    x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Dhaka"),
    x509.NameAttribute(NameOID.LOCALITY_NAME, "Dhaka"),
    x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Iub"),
    x509.NameAttribute(NameOID.COMMON_NAME, "mehediiub.com"),
])
cert = x509.CertificateBuilder().subject_name(
    subject
).issuer_name(
    issuer
).public_key(
    private_key.public_key()
).serial_number(
    x509.random_serial_number()
).not_valid_before(
    datetime.datetime.utcnow()
).not_valid_after(
    datetime.datetime.utcnow() + datetime.timedelta(days=365)
).sign(private_key, hashes.SHA256(), default_backend())

# Serialize Certificate
with open("certificate.pem", "wb") as f:
    f.write(cert.public_bytes(serialization.Encoding.PEM))

# Serialize Private Key
with open("private_key.pem", "wb") as f:
    f.write(private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption()
    ))
===================================================================================================
===================================================================================================
===================================================================================================
===================================================================================================

# Generating signature with X.509
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding

# Load and parse an existing X.509 certificate
with open("certificate.pem", "rb") as f:
    cert_data = f.read()

# Deserialize the certificate
cert = x509.load_pem_x509_certificate(cert_data, default_backend())

# Print some certificate fields
print("Issuer:", cert.issuer)
print("Subject:", cert.subject)
print("Serial Number:", cert.serial_number)
print("Validity Period:", cert.not_valid_before, "to", cert.not_valid_after)

# Load and parse an existing private key (used for demonstration)
with open("private_key.pem", "rb") as f:
    private_key_data = f.read()

private_key = serialization.load_pem_private_key(
    private_key_data,
    password= "**********"
    backend=default_backend()
)

# Sign some data with the private key (this is just an example)

message = b"some data to sign"
signature = private_key.sign(
    message,
    padding.PSS(
        mgf=padding.MGF1(hashes.SHA256()),
        salt_length=padding.PSS.MAX_LENGTH
    ),
    hashes.SHA256()
)


===================================================================================================
===================================================================================================
===================================================================================================
===================================================================================================

# Verifying Signature
with open("certificate.pem", "rb") as f:
    cert_data = f.read()
cert = x509.load_pem_x509_certificate(cert_data, default_backend())
public_key = cert.public_key()

received_signature = signature
try:
    public_key.verify(
        received_signature,
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    print("Signature is valid.")

except:
    print("Signature is invalid.")
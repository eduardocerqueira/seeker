#date: 2025-01-30T16:54:42Z
#url: https://api.github.com/gists/74172a761e604d875a59774189aafb68
#owner: https://api.github.com/users/eugene1g

FROM docker.io/library/almalinux:latest


# LibreOffice version - find the latest at https://downloadarchive.documentfoundation.org/libreoffice/old/latest/rpm/x86_64/
ARG LIBREOFFICE_VERSION=25.2.0.3


ENV LIBREOFFICE_VERSION=${LIBREOFFICE_VERSION}

# Install all dependencies for LibreOffice
RUN dnf install -y java-21-openjdk libXinerama cairo cups

# Download and install LibreOffice
RUN curl -L \
    "https://downloadarchive.documentfoundation.org/libreoffice/old/$LIBREOFFICE_VERSION/rpm/x86_64/LibreOffice_${LIBREOFFICE_VERSION}_Linux_x86-64_rpm.tar.gz" \
    -o libreoffice.tar.gz && \
    tar xzf libreoffice.tar.gz && \
    dnf install -y LibreOffice_${LIBREOFFICE_VERSION}_Linux_x86-64_rpm/RPMS/*.rpm && \
    rm -rf libreoffice.tar.gz LibreOffice_${LIBREOFFICE_VERSION}_Linux_x86-64_rpm && \
    ln -sf $(ls -d /opt/libreoffice*/program/soffice | head -n1) /usr/local/bin/soffice

# Create a working directory for file operations
WORKDIR /documents

COPY <<'EOF' /usr/local/bin/convert.sh
#!/bin/bash
set -e
cat - > "${1:-input_file}"
soffice --headless --convert-to pdf:writer_pdf_Export --outdir /documents "${1:-input_file}"
cat "${1:-input_file}.pdf"
EOF

RUN chmod a+x /usr/local/bin/convert.sh


ENTRYPOINT ["/usr/local/bin/convert.sh"]



# docker build --platform=linux/amd64 -t mylibrepdf .
# echo "docker run --rm -i mylibrepdf" > ~/bin/topdf && chmod +x ~/bin/topdf
# cat Document.docx | topdf > Document.pdf
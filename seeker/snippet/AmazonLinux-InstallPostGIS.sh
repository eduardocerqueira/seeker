#date: 2023-10-20T16:57:52Z
#url: https://api.github.com/gists/0a1e5f7292e01b576161f05874e0fc87
#owner: https://api.github.com/users/yehiaa

#!/bin/bash
# Script to install PostgreSQL and PostGIS on a fresh Amazon Linux instance

# Installing from source:
# - GEOS 
# GEOS 3.10+ requires CMake 3+, not readily available on Amazon Linux 2.
GEOSVER=3.9.2
GEOSURL=http://download.osgeo.org/geos/geos-${GEOSVER}.tar.bz2

# - PROJ (GDAL requires 6+; 6.2.1 is the last to use SQLite 3.7; 6.2 had build issues, so 6.1.1)
PROJVER=6.1.1
PROJURL=https://download.osgeo.org/proj/proj-${PROJVER}.tar.gz

# - GDAL 
GDALVER=3.4.3
GDALURL=https://github.com/OSGeo/gdal/releases/download/v${GDALVER}/gdal-${GDALVER}.tar.gz

# - PostGIS 
POSTGISVER=3.2.1
POSTGISURL=https://download.osgeo.org/postgis/source/postgis-${POSTGISVER}.tar.gz

set -e

sudo amazon-linux-extras install postgresql13 vim epel -y
sudo yum-config-manager --enable epel -y
sudo yum update -y
sudo yum install -y make automake cmake gcc gcc-c++ libcurl-devel proj-devel pcre-devel autoconf automake libxml2-devel libjpeg-turbo-static libjpeg-turbo-devel sqlite-devel
sudo yum install -y clang llvm
sudo yum install -y postgresql-server postgresql-server-devel

############################
# Install GEOS from Source #
############################
curl -O $GEOSURL
tar xvjf geos-${GEOSVER}.tar.bz2
rm -f geos-${GEOSVER}.tar.bz2
cd geos-${GEOSVER}
./configure
make
sudo make install
cd

############################
# Install PROJ from Source #
############################
curl -L -O $PROJURL
tar xvzf proj-${PROJVER}.tar.gz
rm -f proj-${PROJVER}.tar.gz
cd proj-${PROJVER}
./configure
make
sudo make install
cd

############################
# Install GDAL from Source #
############################
curl -L -O $GDALURL
tar xvzf gdal-${GDALVER}.tar.gz
rm -f gdal-${GDALVER}.tar.gz
cd gdal-${GDALVER}
./configure \
    --prefix=${PREFIX} \
    --with-geos \
    --with-proj=/usr/local \
    --with-geotiff=internal \
    --with-hide-internal-symbols \
    --with-libtiff=internal \
    --with-libz=internal \
    --with-threads \
    --without-bsb \
    --without-cfitsio \
    --without-cryptopp \
    --with-curl \
    --without-dwgdirect \
    --without-ecw \
    --without-expat \
    --without-fme \
    --without-freexl \
    --without-gif \
    --without-gif \
    --without-gnm \
    --without-grass \
    --without-grib \
    --without-hdf4 \
    --without-hdf5 \
    --without-idb \
    --without-ingres \
    --without-jasper \
    --without-jp2mrsid \
    --with-jpeg=internal \
    --without-kakadu \
    --without-libgrass \
    --without-libkml \
    --without-libtool \
    --without-mrf \
    --without-mrsid \
    --without-mysql \
    --without-netcdf \
    --without-odbc \
    --without-ogdi \
    --without-openjpeg \
    --without-pcidsk \
    --without-pcraster \
    --with-pcre \
    --without-perl \
    --with-pg \
    --without-php \
    --with-png=internal \
    --without-python \
    --without-qhull \
    --without-sde \
    --without-sqlite3 \
    --without-webp \
    --with-xerces \
    --with-xml2
make
sudo make install
cd
 
###################################
# Install PostGIS from source #
###################################
curl -O $POSTGISURL
tar xvzf postgis-${POSTGISVER}.tar.gz
rm -f postgis-${POSTGISVER}.tar.gz
cd postgis-${POSTGISVER}
./configure --with-address-standardizer --without-protobuf
make
sudo make install
cd

###################
# Final Prep Work #
###################
sudo ln -s /usr/local/lib/libgeos_c.so.1 /usr/lib64/pgsql/libgeos_c.so.1
sudo sh -c 'echo /usr/local/lib > /etc/ld.so.conf.d/postgresql.conf'
sudo sh -c 'echo /usr/lib64/pgsql >> /etc/ld.so.conf.d/postgresql.conf'
sudo ldconfig -v

export PGHOME=/var/lib/pgsql/data/
sudo su postgres -c "pg_ctl -D $PGHOME initdb"

sudo systemctl enable postgresql
sudo systemctl start postgresql

echo "
Your system is now running PostgreSQL with PostGIS. 
You should now run "aws configure" to set up the AWS CLI.   
Afterwards, you should stop this instance and create an AMI. 
"

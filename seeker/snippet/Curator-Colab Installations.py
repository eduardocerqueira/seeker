#date: 2022-10-17T17:17:21Z
#url: https://api.github.com/gists/6f0cdb7837ebf789f1cfc9d12c19af6f
#owner: https://api.github.com/users/Riazone

## Run This Cell for Colab
!pip install yahoo_fin
!pip install mplfinance
!pip install pycaret
!wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
!tar -xzvf ta-lib-0.4.0-src.tar.gz
%cd ta-lib
!./configure --prefix=/usr
!make
!make install
!pip install Ta-Lib
import talib
! pip install pytictoc
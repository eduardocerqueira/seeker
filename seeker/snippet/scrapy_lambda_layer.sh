#date: 2022-01-19T17:07:11Z
#url: https://api.github.com/gists/84788d91c5a710341254762f9598af79
#owner: https://api.github.com/users/TheRockStarDBA

REGION=eu-west-1
VER=1.7.3
RUNTIME=python3.7

docker run -v $(pwd):/out -it lambci/lambda:build-$RUNTIME \
    pip install scrapy==$VER -t /out/build/scrapy/python

cd build/scrapy
zip -r ../../scrapy.zip python/
cd ../..

aws lambda publish-layer-version \
    --layer-name Scrapy \
    --region $REGION \
    --description $VER \
    --zip-file fileb://scrapy.zip \
    --compatible-runtimes $RUNTIME

rm -rf build *.zip

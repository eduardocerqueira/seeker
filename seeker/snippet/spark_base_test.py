#date: 2022-03-22T17:00:31Z
#url: https://api.github.com/gists/34a077be0dd2cb0dfeb6a7dced341b74
#owner: https://api.github.com/users/juanmiguelaltube

import unittest

import pytest
from pyspark import SparkConf
from pyspark.sql import SparkSession


class SparkBaseTest(unittest.TestCase):

    def __init__(self, methodName: str) -> None:
        super(SparkBaseTest, self).__init__(methodName=methodName)

    @pytest.fixture(autouse=True)
    def setup(self):
        conf = (SparkConf()
                # .set("spark.sql.legacy.timeParserPolicy", "LEGACY")
                )
        self.spark = (SparkSession
                      .builder
                      .config(conf=conf)
                      .master('local[4]')
                      .appName("base-test")
                      .getOrCreate())

        self.spark.sparkContext.setLogLevel("ERROR")

    def teardown(self):
        self.spark.stop()

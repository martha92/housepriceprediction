import json
import requests
import sys

assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
from zipfile import ZipFile
from pyspark.sql import SparkSession, types
from io import *
import pandas as pd
from urllib.request import *

spark = SparkSession.builder.appName('Load Tourism Data').getOrCreate()

touristinfo_schema = types.StructType([
    types.StructField('REF_DATE', types.StringType(), True),
    types.StructField('GEO', types.StringType(), True),
    types.StructField('DGUID', types.StringType(), True),
    types.StructField('Total_international_travellers', types.StringType(), True),
    types.StructField('Total_Canadian_residents', types.StringType(), True), ])


def download_extract_zip(url):
    """
    Download a ZIP file and extract its contents in memory
    yields (filename, file-like object) pairs
    """
    response = requests.get(url)
    with ZipFile(BytesIO(response.content)) as thezip:
        for zipinfo in thezip.infolist():
            with thezip.open(zipinfo) as thefile:
                df = pd.read_csv(thefile)
                return (df)


def loadTouristInfo():
    # PRODUCT ID FOR TOURSIM INFO.
    productId = "24100041"
    response = requests.get("https://www150.statcan.gc.ca/t1/wds/rest/getFullTableDownloadCSV/" + productId + "/en")
    jdata = json.loads(response.text)
    zipUrl = jdata['object']
    pdDF = download_extract_zip(zipUrl)
    new_df = pdDF.loc[
        pdDF['Traveller characteristics'].isin(['Total international travellers', 'Total Canadian residents'])]
    transposeDF = new_df.pivot_table(index=['REF_DATE', 'GEO', 'DGUID'], columns='Traveller characteristics',
                                     values='VALUE').reset_index(['REF_DATE', 'GEO', 'DGUID'])
    tourism_df = spark.createDataFrame(transposeDF, schema=touristinfo_schema).createOrReplaceTempView("tourist_data")
    avg_tourist_data = spark.sql("SELECT GEO, REF_DATE, DGUID,'Persons' as uom_tourist, 'unit' as scalar_tourist, \
    Total_international_travellers as international_tourism, Total_Canadian_residents as domestic_tourism FROM tourist_data")
    return avg_tourist_data

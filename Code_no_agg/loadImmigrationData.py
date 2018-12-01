
import json
import requests
import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
from zipfile import ZipFile
from pyspark.sql import SparkSession, types
from io import *
import pandas as pd
from urllib.request import *
spark = SparkSession.builder.master("local[*]").config("spark.executor.memory", "70g").config("spark.driver.memory", "50g").config("spark.memory.offHeap.enabled",True).config("spark.memory.offHeap.size","32g").config("spark.driver.maxResultSize","10g").appName("Load Labour Force Data").getOrCreate()

immigration_schema = types.StructType([
	types.StructField('REF_DATE', types.StringType(), True),
	types.StructField('GEO', types.StringType(), True),
    types.StructField('DGUID', types.StringType(), True),
    types.StructField('In_migrants', types.IntegerType(), True),
    types.StructField('Out_migrants', types.IntegerType(), True),])

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

def loadImmigrationData():
	#PRODUCT ID FOR .
    productId = "17100020"
    response = requests.get("https://www150.statcan.gc.ca/t1/wds/rest/getFullTableDownloadCSV/"+productId + "/en")
    jdata = json.loads(response.text)
    zipUrl = jdata['object']
    pdDF = download_extract_zip(zipUrl)
    transposeDF = pdDF.pivot_table(index = ['REF_DATE','GEO','DGUID'], columns='Interprovincial migration', values='VALUE').reset_index(['REF_DATE','GEO','DGUID'])
    immigration_df = spark.createDataFrame(transposeDF,schema=immigration_schema).createOrReplaceTempView("immigration_info")
    avg_per_province = spark.sql("SELECT GEO, REF_DATE, DGUID, 'Persons' as uom_imm, 'units' as scalar_imm, In_migrants, Out_migrants FROM immigration_info")
    return avg_per_province


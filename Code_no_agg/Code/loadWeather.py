import sys

assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+
from pyspark.sql import SparkSession, types, functions as F
from urllib.request import *
import requests, json
from io import *
import pandas as pd
from zipfile import ZipFile
from pyspark.sql.functions import input_file_name

spark = SparkSession.builder.appName('Load Weather Data').getOrCreate()

weather_schema = types.StructType([
    types.StructField('REF_DATE', types.StringType(), True),
    types.StructField('Year', types.StringType(), True),
    types.StructField('Month', types.StringType(), True),
    types.StructField('Mean_Max_Temp', types.StringType(), True),
    types.StructField('Mean_Max_Temp_Flag', types.StringType(), True),
    types.StructField('Mean_Min_Temp', types.StringType(), True),
    types.StructField('Mean_Min_Temp_Flag', types.StringType(), True),
    types.StructField('Mean_Temp', types.StringType(), True),
    types.StructField('Mean_Temp_Flag', types.StringType(), True),
    types.StructField('Extr_Max_Temp', types.StringType(), True),
    types.StructField('Extr_Max_Temp_Flag', types.StringType(), True),
    types.StructField('Extr_Min_Temp', types.StringType(), True),
    types.StructField('Extr_Min_Temp_Flag', types.StringType(), True),
    types.StructField('Total_Rain', types.StringType(), True),
    types.StructField('Total_Rain_Flag', types.StringType(), True),
    types.StructField('Total_Snow', types.StringType(), True),
    types.StructField('Total_Snow_Flag', types.StringType(), True),
    types.StructField('Total_Precip', types.StringType(), True),
    types.StructField('Total_Precip_Flag', types.StringType(), True),
    types.StructField('Snow_Grnd_Last_Day', types.StringType(), True),
    types.StructField('Snow_Grnd_Last_Day_Flag', types.StringType(), True),
    types.StructField('Dir_of_Max_Gust', types.StringType(), True),
    types.StructField('Dir_of_Max_Gust_Flag', types.StringType(), True),
    types.StructField('Spd_of_Max_Gust', types.StringType(), True),
    types.StructField('Spd of Max Gust_Flag', types.StringType(), True), ])


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


def get_dguid():
    productId = "18100205"
    response = requests.get("https://www150.statcan.gc.ca/t1/wds/rest/getFullTableDownloadCSV/" + productId + "/en")
    jdata = json.loads(response.text)
    zipUrl = jdata['object']
    pdDF = download_extract_zip(zipUrl)
    transposeDF = pdDF.pivot_table(index=['REF_DATE', 'GEO', 'DGUID'], columns='New housing price indexes',
                                   values='VALUE').reset_index(['REF_DATE', 'GEO', 'DGUID'])
    land_schema = types.StructType([
        types.StructField('REF_DATE', types.StringType(), True),
        types.StructField('GEO', types.StringType(), True),
        types.StructField('DGUID', types.StringType(), True),
        types.StructField('House_only', types.StringType(), True),
        types.StructField('Land_only', types.StringType(), True),
        types.StructField('Total_house_land', types.StringType(), True)])
    return spark.createDataFrame(transposeDF, schema=land_schema).select('GEO', 'DGUID').drop_duplicates()


def loadWeatherData():
    weather = spark.read.csv("Other_sources/weather", schema=weather_schema)
    weather.withColumn("input_file", input_file_name()).createOrReplaceTempView("weather_info")
    weather_info = spark.sql("SELECT REF_DATE, Mean_Max_Temp, Mean_Min_Temp, Mean_Temp, Total_Rain, Total_Snow, \
    SUBSTR(substr(input_file, - instr(reverse(input_file), '/') + 1) , 0, \
    INSTR(substr(input_file, - instr(reverse(input_file), '/') + 1) , '_')-1) as province FROM weather_info")
    df1 = weather_info.select(F.trim(F.regexp_replace(F.col('province'), '%20', ' ')).alias('GEO'), 'REF_DATE', 'Mean_Max_Temp', \
                                     'Mean_Min_Temp', 'Mean_Temp', 'Total_Rain', 'Total_Snow').drop_duplicates()
    dguid = get_dguid()
    return df1.join(dguid, ['GEO'], 'inner')


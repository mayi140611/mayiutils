#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: data_explore.py
@time: 2019-05-06 11:06
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == '__main__':
    mode = 1
    if mode == 2:
        """
        看原始数据
        """
        df = pd.read_parquet(
            '/Users/luoyonggui/Documents/datasets/work/2/register.parquet/part-00000-9cbbf3ef-976d-4fa4-8926-4da5700f1d67-c000.snappy.parquet')
        print(df.head())
        print(df.shape)  # (4388, 61)
        df.info()
        """
RangeIndex: 4388 entries, 0 to 4387
Data columns (total 61 columns):
registerNo                   4388 non-null object
hospNo                       4388 non-null object
hospName                     4388 non-null object
perNo                        4388 non-null object
perName                      4388 non-null object
nth_hospital_care            4388 non-null object
sex                          4388 non-null object
zoneName                     4388 non-null object
insuranceNo                  4388 non-null object
medicalType                  63 non-null object
inHospDate                   4388 non-null object
outHospDate                  4388 non-null object
hospDays                     4388 non-null object
settlementDate               4388 non-null object
outHospReason                4388 non-null object
inHospDiagnosisCode          63 non-null object
inHospDiagnosisName          63 non-null object
outHospDiagnosisCode         4388 non-null object
outHospDiagnosisName         4388 non-null object
outHospSecondaryCode         0 non-null object
outHospSecondaryName         0 non-null object
diagnosisDesc                4388 non-null object
bedNo                        4388 non-null object
doctorCode                   63 non-null object
doctorName                   63 non-null object
deptCode                     3945 non-null object
deptName                     3958 non-null object
hospLevel                    4388 non-null object
triggerBy                    4388 non-null object
perType                      4388 non-null object
visit_type                   4388 non-null object
scenarioDesc                 4388 non-null object
scenarioNo                   4388 non-null object
groupNo                      4388 non-null object
groupNos                     4388 non-null object
scenarioNos                  4388 non-null object
registerNumber               4385 non-null object
month                        4388 non-null object
timeScope                    4388 non-null object
settlementSeason             4388 non-null object
zoneId                       4388 non-null object
is24HrBackHosp               4388 non-null object
isOffSiteMedicalTreatment    4388 non-null object
score                        4388 non-null object
status                       4388 non-null int32
institutionOfTheInsured      0 non-null object
is_happen_after_dead         4388 non-null object
ageScope                     1230 non-null object
age                          1230 non-null float64
nth_no_drug                  769 non-null float64
billSum                      4388 non-null float64
billApply                    4388 non-null float64
billSelf                     4388 non-null float64
rs_overallPay                4388 non-null float64
accountPay                   4388 non-null float64
rescuePay                    4388 non-null float64
amount                       4388 non-null float64
drug_ratio                   4388 non-null float64
operation_ratio              4388 non-null float64
inspection_ratio             4388 non-null float64
assisted_ratio               4388 non-null float64
dtypes: float64(13), int32(1), object(47)
memory usage: 2.0+ MB
        """
    if mode == 1:
        """
        看提取完特征的数据
        https://cf.leapstack.cn/pages/viewpage.action?pageId=23954878
        """
        df = pd.read_parquet('/Users/luoyonggui/Documents/datasets/work/2/feature/part-00000-82598d3a-556e-4ce5-948e-29305d9eca5e-c000.snappy.parquet')
        # print(df.head())
        # print(df.shape)#(516392, 15)
        # df.info()
        """
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 516392 entries, 0 to 516391
Data columns (total 15 columns):
registerNo  就诊流水号           516392 non-null object
perICDCount          516392 non-null float64
perICDMonthCount     516392 non-null float64
regDiagRatio         516392 non-null float64
sexFeature           516392 non-null float64
sexFeeFeature        516392 non-null float64
ageFeature           516392 non-null float64
ageFeeFeature        516392 non-null float64
feeFeature           516392 non-null float64
icdFeature           516392 non-null float64
icdMonthFeature      516392 non-null float64
medSex               516392 non-null float64
medAge               516392 non-null float64
hospMonthMedRatio    516392 non-null float64
hospMedRatio         516392 non-null float64
dtypes: float64(14), object(1)
memory usage: 59.1+ MB
        """
        # print(df.describe())
        """
         perICDCount  perICDMonthCount  ...  hospMonthMedRatio  hospMedRatio
count  516392.000000     516392.000000  ...       5.163920e+05  5.163920e+05
mean        1.450886          1.103292  ...       5.434508e-01  8.040084e+00
std         0.751607          0.332364  ...       2.631941e+00  3.729968e+01
min         1.000000          1.000000  ...       1.210242e-08  1.210242e-08
25%         1.000000          1.000000  ...       2.705159e-03  9.949308e-02
50%         1.000000          1.000000  ...       1.054688e-01  1.000000e+00
75%         2.000000          1.000000  ...       1.000000e+00  2.000000e+00
max         9.000000          5.000000  ...       1.275220e+02  6.280189e+02

[8 rows x 14 columns]
        """
        # sns.pairplot(df.iloc[:1000, 1:])
        # plt.show()
        sns.heatmap(df.iloc[:100, 1:].corr())
        plt.show()

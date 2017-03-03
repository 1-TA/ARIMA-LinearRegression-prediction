import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pylab as plt
import matplotlib.patches as mpatches
from datetime import datetime
from arch.unitroot import ADF
from arch.unitroot import KPSS
from sklearn.metrics import r2_score
from sklearn import linear_model

class ArimaMethod:

    def __init__(self, dtConv, source, datacolumnName, endDate, predictionSteps, yValue):
        dateparse = lambda dates: pd.datetime.strptime(str(dates), dtConv) 
        l = list(filter(lambda x: x > 10 and x < 80, l))
        self.originalData = pd.read_csv(source,  index_col = datacolumnName, date_parser=dateparse, infer_datetime_format=True)
        self.originalData = self.originalData[:endDate]
        self.predictionSteps = predictionSteps
        self.yValue =  yValue # 2016/10/28 'TotalRef'
        self.LastYValValue = self.originalData[self.yValue].ix[len(self.originalData)-1]
        NewPatientGrowth = self.calculate_growth(self.originalData[self.yValue])
        NewPatientGrowth.insert(0, 0)
        self.originalData[self.yValue] = NewPatientGrowth
        print(len(self.originalData[self.yValue]))
    
    def setLagDifference(self, lags):
        self.lags = lags

    def getFloatValue(self, fValue):
        a = np.array(fValue)
        try:
            if np.float(tuple(a.flat)[0]):
                return float(tuple(a.flat)[0])
        except:
            print('Exception caught')

    def getLagDifference(self):
        tempdata = pd.Series(self.originalData[self.yValue], copy = True )
        #KPSS
        arch_kpss = KPSS(self.originalData[self.yValue])
        if  arch_kpss.pvalue < 0.05 :
            tempdata = tempdata - tempdata.shift()
            tempdata.dropna(inplace = True)
            arch_kpss = KPSS(tempdata)
            if arch_kpss.pvalue < 0.05 : 
                return -1 #not stationary after 1 lag diff
            else:
                return 1
        else:
            return 0

    def FindPQorder(self): #finding the best order p and q by analyzing minimum AIC
        _P = -1
        _Q = -1
        minAIC = 99999999
        for p in range(4, -1, -1):
            for q in range(4,-1, -1):
                try:
                    model = ARIMA(self.originalData[self.yValue], order=(p, self.lags, q))#changed the dataset 10/24 9:51 Number of Tractor Sold
                    model.endog = model.endog.astype(float)
                    results_AR = model.fit(disp=-1)
                    if(minAIC > results_AR.aic):
                        minAIC = results_AR.aic
                        _P = p
                        _Q = q
                except:
                    continue
        print('Values of P and Q are : ', _P, _Q)
        return _P,_Q

    def calculate_growth(self, data):
        temp = []
        for a in range(1,len(data)):
            gnum = (data.values[a] - data.values[a - 1])/data.values[a-1]
            temp.append(gnum)    
        return temp


    def getArimaModel(self, p, q):
        model = ARIMA(self.originalData[self.yValue], order=(p, self.lags, q))        
        return model

    def getArimaResult(self, p , q):
        model = self.getArimaModel(p,q)
        model.endog = model.endog.astype(float)
        results_AR = model.fit(disp=-1)
        return results_AR

    def realHistoryForcast(self, arimaResultCLS):
        forcastedArray = arimaResultCLS.forecast(steps = self.predictionSteps, alpha=0.05)
        return forcastedArray


    def outlinerDetection(self): 
        q25, q75 = np.percentile(self.originalData[self.yValue], [25,75])
        iqr = q75 - q25
        HCutt = q75 + 1.5 * iqr
        LCutt = q25 - 1.5 * iqr
        LoopLen = len(self.originalData[self.yValue])
        for i in range(0,LoopLen): 
            if self.originalData[self.yValue].values[i] > HCutt or self.originalData[self.yValue].values[i] < LCutt :
                self.originalData[self.yValue].values[i] = (self.originalData[self.yValue].values[i+1] + self.originalData[self.yValue].values[i-1])/2 #storing the index of outliers
        
    def getSeriesLastValue(self):
        return self.LastYValValue

    def getForcastbyGrowth(self, growth):
        lastValue = self.getSeriesLastValue()
        forcast = (growth[0] * lastValue) + lastValue
        self.LastYValValue = int(forcast)
        return forcast
       
    def getR2Score(self, fittedValues):
        r2 = r2_score(self.originalData[self.yValue], fittedValues)
        return r2

class RegressionMethod:
    
    def __init__(self, dtConv, source, datecolumnName, var_indept, var_dep, endDate):
        dateparse = lambda dates: pd.datetime.strptime(str(dates), dtConv) 
        self.originalData = pd.read_csv(source,  index_col = datecolumnName, date_parser=dateparse, infer_datetime_format=True)
        self.originalData = self.originalData[:endDate]
        self.YCol = var_dep
        self.XCol = var_indept
        self.LastYValValue = self.originalData[self.YCol].ix[len(self.originalData)-1]
        
        NewPatientGrowth = self.calculate_growth(self.originalData[self.YCol])
        NewPatientGrowth.insert(0, np.nan)
        ProcedureGrowth = self.calculate_growth(self.originalData[self.XCol])
        ProcedureGrowth.insert(0, np.nan)
        
        self.YCol = 'NewY'
        self.XCol = 'NewX'
        self.originalData[self.YCol] = NewPatientGrowth
        self.originalData[self.XCol] = ProcedureGrowth
        self.originalData.dropna(how = 'any', inplace = True)

    def plotRegressionModel(self, regr):
        plt.scatter(self.X_Train, self.Y_Train , color = 'black')
        plt.plot(self.X_Train, regr.predict(self.X_Train), color='blue')
        plt.show()

    def setValue2Predict(self, val):
        self.valuetopredict = val

    def calculate_growth(self, data):
        temp = []
        for a in range(1,len(data)):
            gnum = (data.values[a] - data.values[a - 1])/data.values[a-1]
            temp.append(gnum)    
        return temp
    
    def loadTrainTest(self):
        data_len = len(self.originalData)
        TT = np.absolute(0.05 * data_len).astype(int)
        print('number of tests involved', TT)
        self.X_Train = pd.Series(self.originalData[self.XCol][:data_len-TT]).reshape(data_len-TT,1)
        self.X_Test  = pd.Series(self.originalData[self.XCol][data_len-TT:]).reshape(TT,1)
        self.Y_Train = pd.Series(self.originalData[self.YCol][:data_len-TT]).reshape(data_len-TT,1)
        self.Y_Test  = pd.Series(self.originalData[self.YCol][data_len-TT:]).reshape(TT,1)
              
    
    def loadLinearReg(self):
        val2predict = self.valuetopredict
        regr = linear_model.LinearRegression()
        regr.fit(self.X_Train, self.Y_Train)
        #print('Coefficients: \n', regr.coef_)
        #print('Variance score: %.2f' % regr.score(self.X_Test, self.Y_Test))
        prediction = regr.predict(val2predict)
        return regr, prediction

    
#deleting the outliers from the training list
    def outlinerDetection(self): 
        q25, q75 = np.percentile(self.originalData[self.YCol], [25,75])
        iqr = q75 - q25
        HCutt = q75 + 1.5 * iqr
        LCutt = q25 - 1.5 * iqr
        LoopLen = len(self.originalData[self.YCol])
        delList = []
        for i in range(0,LoopLen): 
            if self.originalData[self.YCol].values[i] > HCutt or self.originalData[self.YCol].values[i] < LCutt :
                delList.append(i) #storing the index of outliers
        
        #deleting the outliers 
        self.originalData.drop(self.originalData.index[delList], inplace = True)

    def getR2Score(self, regr):
        regScore = regr.score(self.X_Train, self.Y_Train)
        return regScore

    
    def getForcastbyGrowth(self, growth):
        lastValue = self.LastYValValue
        forcast = (growth * lastValue) + lastValue
        self.LastYValValue = int(forcast)
        return forcast

def main():
    _Source = 'C:/Users/Yekta.Yazdani/Desktop/Data/New Patient Data/d1009/D1009HistoryPatApp.csv' #( , index_col = 'C:/Users/Yekta.Yazdani/Downloads/Tractor-Sales.csv'
    _Dateformat = '%m/%d/%Y' #'%b-%y' '%Y%m'
    _DateColumnName = 'Date' #'Month-Year'
    _EndDate = '2016-07' #'2014-10' 
    _PredictionSteps = 1
    _YValueData = 'NewPatCnt' #'Number of Tractor Sold'
    _XCol = 'Sum of Appt'
    ARM = ArimaMethod(_Dateformat,_Source, _DateColumnName, _EndDate, _PredictionSteps, _YValueData)
    ARM.outlinerDetection()
    _Lags = ARM.getLagDifference()
    ARM.setLagDifference(_Lags)
    _P,_Q = ARM.FindPQorder()
    arimaResult = ARM.getArimaResult(_P,_Q)
    arimaResult.plot_predict(0, '2017', dynamic=False, plot_insample=True)
    plt.show()
    ForcastGrowth = ARM.realHistoryForcast(arimaResult)   
    ArimaR2 = ARM.getR2Score(arimaResult.fittedvalues)
#    print('The predicted values for the next ', _PredictionSteps , ' periods are:\n', ARM.getForcastbyGrowth(ForcastGrowth[0]))

    REG = RegressionMethod(_Dateformat, _Source, _DateColumnName, _XCol, _YValueData, _EndDate)
    REG.outlinerDetection()
    REG.loadTrainTest()
    REG.setValue2Predict(0.0945945945945946)
    regr, prediction = REG.loadLinearReg()
    LinerRegR2 = REG.getR2Score(regr)
    REG.plotRegressionModel(regr)
    
    if(ArimaR2 > LinerRegR2):
        print('Arima Advantage, ArimaR2 : ', ArimaR2, ' Linear R2: ', LinerRegR2)    
        print("ARIMA Performed a Better Forcast, Number of new Patients for next period would be " , int(ARM.getForcastbyGrowth(ForcastGrowth)))
        print("Linear Regression Performed a Better Forcast, Number of new Patients for next period would be " , int(REG.getForcastbyGrowth(prediction)))

    else:
        print('Linear Advantage: ', ArimaR2, ' Linear R2: ', LinerRegR2)
        print("ARIMA Performed a Better Forcast, Number of new Patients for next period would be " , int(ARM.getForcastbyGrowth(ForcastGrowth)))
        print("Linear Regression Performed a Better Forcast, Number of new Patients for next period would be " , int(REG.getForcastbyGrowth(prediction)))
if __name__ == '__main__':
    main()

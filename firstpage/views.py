from django.shortcuts import render
from django.http import HttpResponse
#we will import pandas
import pandas as pd

import urllib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
# Create your views here.

#this will start first page
from sklearn import pipeline,preprocessing,metrics,model_selection,ensemble
#from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from io import StringIO
from io import BytesIO
import base64
import statsmodels.api as sm
reg_fit = 1

def index(request):
    return render(request, 'index.html', {'imp':False})

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
class ExponentialSaturation(BaseEstimator, TransformerMixin):
    def __init__(self, a=1.):
        self.a = a
        
    def fit(self, X, y=None):
        X = check_array(X)
        self._check_n_features(X, reset=True) # from BaseEstimator
        return self
    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False) # from BaseEstimator
        return 1 - np.exp(-self.a*X)

from scipy.signal import convolve2d
import numpy as np
class ExponentialCarryover(BaseEstimator, TransformerMixin):
    def __init__(self, strength=0.5, length=1):
        self.strength = strength
        self.length = length
    def fit(self, X, y=None):
        X = check_array(X)
        self._check_n_features(X, reset=True)
        self.sliding_window_ = (
            self.strength ** np.arange(self.length + 1)
        ).reshape(-1, 1)
        return self
    def transform(self, X: np.ndarray):
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)
        convolution = convolve2d(X, self.sliding_window_)
        if self.length > 0:
            convolution = convolution[: -self.length]
        return convolution    

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
adstock = ColumnTransformer(
    [
     ('tv_pipe', Pipeline([
                           ('carryover', ExponentialCarryover()),
                           ('saturation', ExponentialSaturation())
     ]), ['cyl']),
     ('radio_pipe', Pipeline([
                           ('carryover', ExponentialCarryover()),
                           ('saturation', ExponentialSaturation())
     ]), ['disp']),
     ('social_media_pipe', Pipeline([
                           ('carryover', ExponentialCarryover()),
                           ('saturation', ExponentialSaturation())
     ]), ['wt']),
         ],
    remainder='passthrough'
)
model_adstock = Pipeline([
                  ('adstock', adstock),
                  ('regression', LinearRegression())
])


from optuna.integration import OptunaSearchCV
from optuna.distributions import UniformDistribution, IntUniformDistribution
tuned_model = OptunaSearchCV(
    estimator=model_adstock,
    param_distributions={
        'adstock__tv_pipe__carryover__strength': UniformDistribution(0, 1),
        'adstock__tv_pipe__carryover__length': IntUniformDistribution(0, 6),
        'adstock__tv_pipe__saturation__a': UniformDistribution(0, 0.01),
        'adstock__radio_pipe__carryover__strength': UniformDistribution(0, 1),
        'adstock__radio_pipe__carryover__length': IntUniformDistribution(0, 6),
        'adstock__radio_pipe__saturation__a': UniformDistribution(0, 0.01),
        'adstock__social_media_pipe__carryover__strength': UniformDistribution(0, 1),
        'adstock__social_media_pipe__carryover__length': IntUniformDistribution(0, 6),
        'adstock__social_media_pipe__saturation__a': UniformDistribution(0, 0.01),
    },
    n_trials=10,
    cv=TimeSeriesSplit(),
    random_state=0
)

def saturation(request):
    x_data,y_data=data()
    tuned_model.fit(x_data.iloc[:,[0,1,2]], y_data)
    value=pd.DataFrame.from_dict(tuned_model.best_params_,orient='index',columns=["value"])
    # applying get_value() function 
    tv_sat_a = value._get_value('adstock__tv_pipe__saturation__a', 'value')
    
    radio_sat_a = value._get_value('adstock__radio_pipe__saturation__a', 'value')

    Social_Media_sat_a = value._get_value('adstock__social_media_pipe__saturation__a', 'value')



    y_axis_TV = 1- np.exp(range(0,1100)*(-tv_sat_a))
    y_axis_radio = 1- np.exp(range(0,1100)*(-radio_sat_a))
    y_axis_Social_Media = 1- np.exp(range(0,1100)*(-Social_Media_sat_a))

    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure

    plt.plot(range(0,1100),y_axis_TV, label='cyl')
    plt.plot(range(0,1100),y_axis_radio, label='disp')
    plt.plot(range(0,1100),y_axis_Social_Media, label='wt')

    plt.legend()
    plt.show()
    
    #fig = plt.gcf()
    buffer = BytesIO()
    plt.savefig(buffer.write(), format='png')
    buffer.seek(0)
    #string = base64.b64encode(buffer.read())
    image_png = buffer.getvalue()
    uri = base64.b64encode(image_png)
    #uri = urllib.parse.quote(string)     
    uri = uri.decode('utf-8')
    buffer.close()
    return render(request, 'mmm.html', {'x':uri})

def imp_features(request):
        uri=imp()
        return render(request, 'mmm.html',{'x':uri})
    
def area_plot(request):
        return render(request, 'mmm.html',{'x':"uri"})    
    
    
#this is user defined function to load the csv data into a  dataframe(name=csv) and to upload it in mysql database
def result(request):
    
    if request.method == "POST":
        file = request.FILES["myFile"]
        csv=pd.read_csv(file)
      
        arr=csv["cyl"]
        sumation={'x':"sum value"}
        y=csv.iloc[:,[0]]
        X=csv.iloc[:,[1,2,4,5]]
        from sklearn.linear_model import LinearRegression
        
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error as mae
        global data
        def data(x_data=X,y_data=y):
            return(x_data,y_data)
        
        global imp
        def imp(x_data=X,y_data=y):
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)
            model = RandomForestRegressor(random_state=1)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            feat_importances = pd.Series(model.feature_importances_, index=x_data.columns)
            x=feat_importances.to_dict()
            plt.bar(list(x.keys()),list(x.values()))
            plt.xlabel("attributes")
            plt.ylabel("importance")
            #feat_importances.nlargest(25).plot(kind='barh',figsize=(10,10))
            #feat_importances_plot = pd.DataFrame(feat_importances).plot(kind = 'barh')
            fig = plt.gcf()
            buffer = BytesIO()
            fig.savefig(buffer, format='png')
            buffer.seek(0)
            string = base64.b64encode(buffer.read())
            uri = urllib.parse.quote(string) 
            return uri

        
        global regression
        def regression(x_pred,x_data = X, y_data=y):
            pipeline_obj=pipeline.Pipeline([("model",LinearRegression())])
            pipeline_obj.fit(x_data,y_data)
            pred = pipeline_obj.predict(x_pred)
            return pred
        
        return render(request, "index.html",{"something":2 , 'x':"uri", 'imp':True})
    else:
        reg_fit = 5        
        return render(request, "index.html")

##this is user defined function to go to the upload html page where is the upload button of csv file
def upload(request):
    return render(request, "upload.html")


def viewdb(request):
    return render(request, 'index.html')

def predictMPG(request):
    if request.method == 'POST':
        temp={}
        temp['cyl']=request.POST.get('cylinderVal')
        temp['disp']=request.POST.get('dispVal')
        #temp['hp']=request.POST.get('hrsPwrVal')
        temp['wt']=request.POST.get('weightVal')
        temp['acc']=request.POST.get('accVal')
        #temp['modyr']=request.POST.get('modelVal')
        #temp['origin']=request.POST.get('originVal')

    testDtaa = pd.DataFrame({'x':temp}).transpose()
    scoreval = regression(testDtaa)
    context={'scoreval':scoreval,'summary':"reg summary"}
    
    
    return render(request, 'result.html',context)

def boxplot(request):
    context={'something':True , 'graph':"BOXPLOT"}
    
    return render(request, 'result.html',context)


def barplot(request):
    context={'something2':True , 'graph2':"barplot"}
    
    return render(request, 'result.html',context)


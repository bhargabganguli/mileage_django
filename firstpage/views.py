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


#this is user defined function to load the csv data into a  dataframe(name=csv)

from django.views.decorators.cache import cache_control
@cache_control(no_cache=True, must_revalidate=True, no_store=True)
def result(request):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error as mae
    if request.method == "POST":
        file = request.FILES["myFile"]
        csv=pd.read_csv(file)
        #csv2 = request.session.get('csv',csv)
        #request.session['csv'] = csv
        size=csv.shape
        #csv=csv.sort_values(by='Date')
        #csv=csv.reset_index()
        #request.session['y'] = csv.iloc[:,[3]]
        #request.session['x'] = csv.iloc[:,[0,1,2]]
        #del request.session['x']
        #del request.session['y']
        #del request.session['x']
        from sklearn.linear_model import LinearRegression
        
        y = csv.iloc[:,[4]]
        X = csv.iloc[:,[1,2,3]]
        x_data,y_data=X,y
        tuned_model.fit(x_data.iloc[:,[0,1,2]], y_data)
        value=pd.DataFrame.from_dict(tuned_model.best_params_,orient='index',columns=["value"])
        # applying get_value() function 
        tv_sat_a = value._get_value('adstock__tv_pipe__saturation__a', 'value')
        radio_sat_a = value._get_value('adstock__radio_pipe__saturation__a', 'value')
        Social_Media_sat_a = value._get_value('adstock__social_media_pipe__saturation__a', 'value')
        
        tv_sat_point = np.log(0.1)/(-tv_sat_a)
        radio_sat_point = np.log(0.1)/(-radio_sat_a)
        Social_Media_sat_point = np.log(0.1)/(-Social_Media_sat_a)
        sat_point = [tv_sat_point,radio_sat_point,Social_Media_sat_point]
        
        y_axis_TV = 1- np.exp(range(0,(int(max(sat_point))+1000))*(-tv_sat_a))
        y_axis_radio = 1- np.exp(range(0,(int(max(sat_point))+1000))*(-radio_sat_a))
        y_axis_Social_Media = 1- np.exp(range(0,(int(max(sat_point))+1000))*(-Social_Media_sat_a))
 
        import matplotlib.pyplot as plt
        from matplotlib.pyplot import figure
        plt.clf()
        plt.plot(range(0,(int(max(sat_point))+1000)),y_axis_TV, label=list(x_data.columns)[0])
        plt.plot(range(0,(int(max(sat_point))+1000)),y_axis_radio, label=list(x_data.columns)[1])
        plt.plot(range(0,(int(max(sat_point))+1000)),y_axis_Social_Media, label=list(x_data.columns)[2])
        plt.legend()
        plt.show()
        #converting plt to bytes to pass in template
        buffer = BytesIO()
        buffer.flush()    
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        uri = base64.b64encode(image_png)
        #uri = urllib.parse.quote(string)     
        uri = uri.decode('utf-8')
        buffer.close()
        
        lr = LinearRegression()
        x_data,y_data=X,y
        lr.fit(x_data,y_data)
        weights = pd.Series(lr.coef_[0],index=x_data.columns)
        base = lr.intercept_[0]
        unadj_contributions = x_data.mul(weights).assign(Base=base)
        adj_contributions = (unadj_contributions.div(unadj_contributions.sum(axis=1), axis=0))
    
        adj_contributions = adj_contributions.mul(np.array(y_data), axis=0)
        # contains all contributions for each day
    
        ax = (adj_contributions[['Base', 'TV', 'Radio', 'Social_Media']].plot.area(figsize=(16, 10),color={'TV':'#AA3142', 'Radio':'#17A589', 'Social_Media':'#9C640C', 'Base':'#1f6f15'},linewidth=1,title='Predicted Sales and Breakdown',ylabel='Sales',xlabel='Date'))
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1],title='Channels', loc="center left",bbox_to_anchor=(1.01, 0.5))
       
        buffer = BytesIO()
        buffer.flush()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.clf()
        image_png = buffer.getvalue()
        uri2 = base64.b64encode(image_png)
        uri2 = uri2.decode('utf-8')
        buffer.close()
    
        # TUNED AREA PLOT
        #tuned_model.fit(x_data.iloc[:,[0,1,2]], y_data)
        adstock_data = pd.DataFrame(tuned_model.best_estimator_.named_steps['adstock'].transform(x_data),columns=x_data.columns,index=x_data.index)
        weights = pd.Series(tuned_model.best_estimator_.named_steps['regression'].coef_[0],index=x_data.columns)
        base = tuned_model.best_estimator_.named_steps['regression'].intercept_
        unadj_contributions = adstock_data.mul(weights).assign(Base=base[0])
        adj_contributions = (unadj_contributions.div(unadj_contributions.sum(axis=1), axis=0).mul(np.array(y_data), axis=0))
        ax = (adj_contributions[['Base', 'Social_Media', 'Radio', 'TV']].plot.area(figsize=(16, 10),color={'TV':'#AA3142', 'Radio':'#17A589', 'Social_Media':'#9C640C', 'Base':'#1f6f15'},linewidth=1,title='Predicted Sales and Breakdown',ylabel='Sales',xlabel='Date'))
    
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1],title='Channels', loc="center left",bbox_to_anchor=(1.01, 0.5))
        
        buffer = BytesIO()
        buffer.flush()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.clf()
        image_png = buffer.getvalue()
        uri3 = base64.b64encode(image_png)  
        uri3 = uri3.decode('utf-8')
        buffer.close()
        
        #IMP feature
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)
        model = RandomForestRegressor(random_state=1)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        feat_importances = pd.Series(model.feature_importances_, index=x_data.columns)
        x=feat_importances.to_dict()
        plt.bar(list(x.keys()),list(x.values()))
        plt.xlabel("attributes")
        plt.ylabel("importance")

        fig = plt.gcf()
        buffer = BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        plt.clf()
        string = base64.b64encode(buffer.read())
        uri4 = urllib.parse.quote(string)         
        
        #carry over
        #tuned_model.fit(x_data.iloc[:,[0,1,2]], y_data)
        value = pd.DataFrame.from_dict(tuned_model.best_params_ , orient='index', columns=["value"] )
        tv_carry_week = int(value._get_value('adstock__tv_pipe__carryover__length', 'value'))
        w=100
        tv_carry_strength = value._get_value('adstock__tv_pipe__carryover__strength', 'value')
        val={0:100,1:0,2:0}
        for i in range(1,tv_carry_week+1):
            val[i] = w*tv_carry_strength
            w=w*tv_carry_strength

        week_no = list(val.keys())
        values = list(val.values())
        plt.bar(week_no, values)
        plt.ylabel('% of carry')
        plt.xlabel('no. of week')
        
        buffer = BytesIO()
        buffer.flush()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.clf()
        image_png = buffer.getvalue()
        uri5 = base64.b64encode(image_png)   
        uri5 = uri5.decode('utf-8')
        buffer.close()
    
        radio_carry_week = int(value._get_value('adstock__radio_pipe__carryover__length', 'value'))
        w=100
        val2={0:100,1:0,2:0}
        radio_carry_strength = value._get_value('adstock__radio_pipe__carryover__strength', 'value')
        for i in range(1,radio_carry_week+1):
            val2[i] = w*radio_carry_strength
            w=w*radio_carry_strength
        week_no = list(val2.keys())
        values = list(val2.values())
        plt.bar(week_no, values)
        plt.ylabel('% of carry')
        plt.xlabel('no. of week')
        
        buffer = BytesIO()
        buffer.flush()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.clf()
        image_png = buffer.getvalue()
        uri6 = base64.b64encode(image_png)   
        uri6 = uri6.decode('utf-8')
        buffer.close()
    
        Social_Media_carry_week = int(value._get_value('adstock__social_media_pipe__carryover__length', 'value'))
        #Social_Media_carry_week = 4
        w=100
        Social_Media_carry_strength = value._get_value('adstock__social_media_pipe__carryover__strength', 'value')
        val3={0:100,1:0,2:0}
        for i in range(1,Social_Media_carry_week+1):
            val3[i] = w*Social_Media_carry_strength
            w=w*Social_Media_carry_strength
        week_no = list(val3.keys())
        values = list(val3.values())
        plt.bar(week_no, values)
        plt.ylabel('% of carry')
        plt.xlabel('no. of week')
        
        buffer = BytesIO()
        buffer.flush()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.clf()
        image_png = buffer.getvalue()
        uri7 = base64.b64encode(image_png)   
        uri7 = uri7.decode('utf-8')
        buffer.close()      

        from docplex.mp.model import Model 
        m = Model(name='Optimization_for_MMM')
        coef = tuned_model.best_estimator_.named_steps['regression'].coef_
        # Variables
        TV = m.integer_var(name='TV')
        Radio = m.integer_var(name='Radio')
        SM = m.integer_var(name='Social_Media')

        # Constraints
        ## On Tv
        TV_non_neg = m.add_constraint(TV >= 1)
        TV_Max = m.add_constraint(TV <= tv_sat_point)
        ## On SM
        #SM_Min = m.add_constraint(SM >= 100)
        SM_Max = m.add_constraint(SM <= 250)
        SM_non_neg = m.add_constraint(SM >= 1)

        ## On Radio
        #Radio_Min = m.add_constraint(Radio >= 120)
        Radio_Max = m.add_constraint(Radio <= 400)
        Radio_non_neg = m.add_constraint(Radio >= 1)

        # Constraints on Total ad spend
        Total_budget_max = m.add_constraint(m.sum([TV + Radio + SM]) <= int(request.POST.get('budget')))

        # Coefficient
        TV_coef = coef[0][0]
        Radio_coef = coef[0][1]
        SM_coef = coef[0][2]
        intercept = 84.68

        # Optimized Budget
        m.maximize(TV*TV_coef + Radio*Radio_coef + SM*SM_coef + intercept)
        sol = m.solve()
        data2 = []
        for v in m.iter_variables():
             data2.append(sol.get_value(v)) 
        #data2 = m.iter_variables()
        frame = pd.DataFrame(data2)
  
        #
        context={"something":2 , 'optimize':data2, 'saturation':uri, 'sat_point':sat_point, 'tv_sat':sat_point[0],'radio_sat':sat_point[1],'Social_Media_sat':sat_point[2],'area_plot':uri2, 'tuned_area_plot':uri3, 'imp_feature':uri4, 'tv_carry':uri5, 'radio_carry':uri6, 'Social_Media_carry':uri7, 'tv_column':X.columns[0], 'radio_column':X.columns[1], 'Social_Media_column':X.columns[2],'imp':True}
        return render(request, "mmm.html",context)
    else:   
        return render(request, "index.html")
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
#x_data,y_data=data()

adstock = ColumnTransformer(
    [
     ('tv_pipe', Pipeline([
                           ('carryover', ExponentialCarryover()),
                           ('saturation', ExponentialSaturation())
     ]), ['TV']),
     ('radio_pipe', Pipeline([
                           ('carryover', ExponentialCarryover()),
                           ('saturation', ExponentialSaturation())
     ]), ['Radio']),
     ('social_media_pipe', Pipeline([
                           ('carryover', ExponentialCarryover()),
                           ('saturation', ExponentialSaturation())
     ]), ['Social_Media']),
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
    n_trials=100,
    cv=TimeSeriesSplit(),
    random_state=0
)

def saturation(request):
    return render(request, 'mmm.html', {'x':"uri"})

def imp_features(request):
    return render(request, 'mmm.html',{'x':"uri"})
    
def area_plot(request):
    return render(request, 'mmm.html',{'x':uri,'y':"uri2"})    
    
    


##this is user defined function to go to the upload html page where is the upload button of csv file
def upload(request):
    return render(request, "upload.html")


def viewdb(request):
    return render(request, 'index.html')

def predictMPG(request):
    context={'scoreval':"scoreval",'summary':"reg summary"}
    return render(request, 'result.html',context)

def boxplot(request):
    context={'something':True , 'graph':"BOXPLOT"}
    return render(request, 'result.html',context)

def barplot(request):
    context={'something2':True , 'graph2':"barplot"}
    return render(request, 'result.html',context)

def carry(request):
    return render(request, 'mmm.html',{'x':uri,'y':uri2,'z':uri3,'carry':True})  

def optimise(request):
    from docplex.mp.model import Model 
    m = Model(name='Optimization_for_MMM')

    # Variables
    TV = m.integer_var(name='TV')
    Radio = m.integer_var(name='Radio')
    SM = m.integer_var(name='Social_Media')

    # Constraints
    ## On Tv
    TV_non_neg = m.add_constraint(TV >= 1)

    ## On SM
    #SM_Min = m.add_constraint(SM >= 100)
    SM_Max = m.add_constraint(SM <= 250)
    SM_non_neg = m.add_constraint(SM >= 1)

    ## On Radio
    #Radio_Min = m.add_constraint(Radio >= 120)
    Radio_Max = m.add_constraint(Radio <= 400)
    Radio_non_neg = m.add_constraint(Radio >= 1)

    # Constraints on Total ad spend
    Total_budget_max = m.add_constraint(m.sum([TV + Radio + SM]) <= int(request.POST.get('budget')))

    # Coefficient
    TV_coef = 3.425
    Radio_coef = 1.07
    SM_coef = 2.367
    intercept = 84.68

    # Optimized Budget
    m.maximize(TV*TV_coef + Radio*Radio_coef + SM*SM_coef + intercept)
    sol = m.solve()
    data2 = []
    for v in m.iter_variables():
        data2.append(sol.get_value(v)) 
    #data2 = m.iter_variables()
    frame = pd.DataFrame(data2)
    return render(request, 'result.html', {'scoreval':x_data.shape, 'summary':data2})

from django.shortcuts import render
from django.http import HttpResponse
#we will import pandas
import pandas as pd
import urllib
import numpy as np
from sklearn.linear_model import LinearRegression
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

def area_plot(request):
    x_data,y_data=data()
    lr = LinearRegression()
    lr.fit(x_data, y_data)
    result = sm.OLS(y_data, x_data).fit()
    df = pd.read_html(result.summary().tables[1].as_html(),header=0,index_col=0)[0]
    a=df['coef']
    #weights = pd.Series(result.params)
    #z=type(lr.coef_)
    base = lr.intercept_

    
    unadj_contributions = x_data.multiply([2,3,4,5]).assign(Base=base)
    """
    adj_contributions = (unadj_contributions.div(unadj_contributions.sum(axis=1), axis=0).mul(y_data, axis=0)) # contains all contributions for each day
    ax = (adj_contributions[['Base', 'cyl', 'disp', 'wt', 'acc']].plot.area(figsize=(16, 10),linewidth=1,title='Predicted Sales and Breakdown',ylabel='Sales',xlabel='Date'))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1],title='Channels', loc="center left",bbox_to_anchor=(1.01, 0.5))
    fig = plt.gcf()
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    string = base64.b64encode(buffer.read())
    uri = urllib.parse.quote(string)     
    """
    return render(request, 'mmm.html', {'x':unadj_contributions})

def imp_features(request):
        uri=imp()
        return render(request, 'mmm.html',{'x':uri})
    
    
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


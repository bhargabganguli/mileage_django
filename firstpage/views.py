from django.shortcuts import render
from django.http import HttpResponse
#we will import pandas
import pandas as pd
# Create your views here.
#import mysql.connector
#from mysql.connector.constants import ClientFlag
#this will start first page
from sklearn import pipeline,preprocessing,metrics,model_selection,ensemble
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
#from io import StringIO
import statsmodels.api as sm
reg_fit = 1
def index(request):
    return render(request, 'index.html')


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
    
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)
        model = RandomForestRegressor(random_state=1)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        feat_importances = pd.Series(model.feature_importances_, index=X.columns)
        #feat_importances.nlargest(25).plot(kind='barh',figsize=(10,10))
        
        global regression
        def regression(x_pred,x_data = X, y_data=y):
            pipeline_obj=pipeline.Pipeline([("model",LinearRegression())])
            pipeline_obj.fit(x_data,y_data)
            pred = pipeline_obj.predict(x_pred)
            return pred
        
        
        """
        import joblib
        joblib.dump(pipeline_obj,'RegModelforMPG4.pkl')
        """
        #joblib.dump(pipeline_obj,'RegModelforMPG4.pkl')
        """
        import sqlalchemy
        from sqlalchemy import create_engine
        my_conn=create_engine("Driver={ODBC Driver 13 for SQL Server};Server=tcp:bhargabbbrio.database.windows.net,1433;Database=bhargab;Uid=bhargab;Pwd={your_password_here};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;")
        
        #mydb = mysql.connector.connect(host="bhargab.mysql.pythonanywhere-services.com", user="bhargab", passwd="Rahara2004", database="bhargab$mileage")
        
        #mycursor = mydb.cursor()
        
        #csv.to_sql(con=mydb, name='mytable2', if_exists='replace')
        #mydb.commit()

 #       mydb = mysql.connector.connect(host="bhargab.mysql.pythonanywhere-services.com", user="bhargab", passwd="Rahara2004", database="bhargab$mileage")
 #       mycursor = mydb.cursor()

 #       mycursor.execute("DROP TABLE IF EXISTS mytable;")

 #       csv.to_sql(con=my_conn,name='mytable',if_exists='append',index=False)
            """
    
        return render(request, "index.html",{"something":2 , 'x':"feat_importances"})
    else:
        reg_fit = 5        
        return render(request, "index.html")

##this is user defined function to go to the upload html page where is the upload button of csv file
def upload(request):
    return render(request, "upload.html")


def viewdb(request):
#    mydb = mysql.connector.connect(host="bhargab.mysql.pythonanywhere-services.com", user="bhargab", passwd="Rahara2004", database="bhargab$mileage")
#    mycursor = mydb.cursor()

#    mycursor.execute("select * from mytable;")
#    result=mycursor.fetchall()

    mpg = []
    cyl = []
    disp= []
    hp = []
    wt = []
    acc = []
    modyr = []
    origin = []
    name = []

    for a_tuple in csv:
        mpg.append(a_tuple[0])

    for a_tuple in csv:
        cyl.append(a_tuple[1])

    for a_tuple in csv:
        disp.append(a_tuple[2])

    for a_tuple in csv:
        hp.append(a_tuple[3])

    for a_tuple in csv:
        wt.append(a_tuple[4])

    for a_tuple in csv:
        acc.append(a_tuple[5])

    for a_tuple in csv:
        modyr.append(a_tuple[6])

    for a_tuple in csv:
        origin.append(a_tuple[7])

    for a_tuple in csv:
        name.append(a_tuple[8])


    mylist = zip( mpg, cyl, disp, hp, wt, acc, modyr, origin, name)
    #DEFINING DICTIONARY TO RENDER IN HTML
    context2 = {
            'mylist': mylist,
            'a':'mileage',
            'b':'cylinder',
        }
    return render(request, 'database.html',context2)

def predictMPG(request):

    """    
    import pandas as pd
    import numpy as np
    from sklearn import pipeline,preprocessing,metrics,model_selection,ensemble
    from sklearn_pandas import DataFrameMapper
    from sklearn.preprocessing import OneHotEncoder
    import statsmodels.api as sm

    #mydb = mysql.connector.connect(host="bhargab.mysql.pythonanywhere-services.com", user="bhargab", passwd="Rahara2004", database="bhargab$mileage")
    #mycursor = mydb.cursor()

    #data=pd.read_sql('select * from mytable',mydb)

    # target and features
    y=data.iloc[:,[0]]
    X=data.iloc[:,[1,2,4,5]]


    # Fit a regression model
    res = sm.OLS(y,X).fit()
    #res = sm.OLS(y.astype(float), X.astype(float)).fit()
    #print(res.summary())


    from sklearn.linear_model import LinearRegression

    pipeline_obj=pipeline.Pipeline([
                ("model",LinearRegression())
    ])



    pipeline_obj.fit(X,y)
    pipeline_obj.predict(X)

    import joblib
    joblib.dump(pipeline_obj,'RegModelforMPG4.pkl')
    #joblib.dump(pipeline_obj,'RegModelforMPG4.pkl')

"""

    #print (request)
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

    """reloadModel=joblib.load('RegModelforMPG4.pkl')
    
    scoreval = reloadModel.predict(testDtaa)[0][0]
    """
  #  rsq = reloadModel.score(X,y)
    #coef = reloadModel.coef_
#    regr=LinearRegression()
#    regr.fit(X,y)
#    coef=regr.coef_
#    scoreval = res.predict(exog=dict(x1=testDtaa))
    #scoreval=2
    
    scoreval = regression(testDtaa)
    context={'scoreval':scoreval,'summary':"reg summary"}
    
    
    return render(request, 'result.html',context)

def boxplot(request):
    
    """
    import matplotlib.pyplot as plt
    from io import StringIO

    mydb = mysql.connector.connect(host="bhargab.mysql.pythonanywhere-services.com", user="bhargab", passwd="Rahara2004", database="bhargab$mileage")
    mycursor = mydb.cursor()

    data=pd.read_sql('select * from mytable',mydb)

    y=data.iloc[:,[0]]
    X=data.iloc[:,[1,2,4,5]]

    fig = plt.figure()
    plt.boxplot(y)
    plt.xlabel("mpg")
    plt.ylabel("frequency")
    plt.title("boxplot")


    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)

    dta = imgdata.getvalue()
    """
    context={'something':True , 'graph':"BOXPLOT"}
    
    return render(request, 'result.html',context)


def barplot(request):
    """
    import matplotlib.pyplot as plt
    from io import StringIO

    mydb = mysql.connector.connect(host="bhargab.mysql.pythonanywhere-services.com", user="bhargab", passwd="Rahara2004", database="bhargab$mileage")
    mycursor = mydb.cursor()

    data=pd.read_sql('select * from mytable',mydb)


    y=data.iloc[:,[0]]
    X=data.iloc[:,[1]]

    fig = plt.figure(figsize = (10, 5))
    plt.hist(y, color ='maroon',width = 0.4)
    plt.xlabel("mpg")
    plt.ylabel("frequency")
    plt.title("histogram")


    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)

    dta = imgdata.getvalue()
    """
    context={'something2':True , 'graph2':"barplot"}
    
    return render(request, 'result.html',context)


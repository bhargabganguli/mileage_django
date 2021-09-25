from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
# Create your views here.

#from sklearn.externals import joblib
import joblib
#from sklearn import preprocessing

reloadModel=joblib.load('./models/RFModelforMPG3.pkl')

def index(request):
    context={'a':'Helloworld!'}
    return render(request, 'index.html',context)
    #return HttpResponse({'a':1})

def predictMPG(request):
    #print (request)
    if request.method == 'POST':
        temp={}
        temp['cyl']=request.POST.get('cylinderVal')
        temp['disp']=request.POST.get('dispVal')
        #temp['hp']=request.POST.get('hrsPwrVal')
        temp['wt']=request.POST.get('weightVal')
        temp['acc']=request.POST.get('accVal')
        temp['modyr']=request.POST.get('modelVal')
        temp['origin']=request.POST.get('originVal')

        print(temp)

    testDtaa = pd.DataFrame({'x':temp}).transpose()
    scoreval = reloadModel.predict(testDtaa)[0]
    context={'scoreval':scoreval}
    #context={'scoreval':'hi'}
    return render(request, 'result.html',context)

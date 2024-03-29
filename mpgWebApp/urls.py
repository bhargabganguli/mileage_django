"""mpgWebApp URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls import url
from firstpage import views
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
urlpatterns = [
    path('admin/', admin.site.urls),
    url('^$',views.index, name='Homepage'),
    url('predictMPG',views.predictMPG,name='PredictMPG'),
    url('MPGdb',views.viewdb,name='PredictMPG'),
    
    url('submit',views.result,name='upload'),
    
    url('upload',views.upload,name='upload'),
    
    url('barplot',views.barplot,name='barplot'),
    
    url('pareto',views.pareto,name='pareto'),
    
    url('boxplot',views.boxplot,name='PredictMPG'),
    
    url('histogram',views.histogram,name='PredictMPG'),
    
    url('scatter',views.scatterplot,name='scatter'),
    
    url('about',views.about,name='about'),
    
    url('game',views.game,name='game'),
    
    url('matchstick',views.matchstick,name='matchstick'),
    
    url('statistics',views.statistics,name='statistics'),
    
    url('ttest',views.ttest,name='ttest'),
    
    url('imp_features',views.imp_features,name='impfeat'),
    url('saturation',views.saturation,name='saturation'),
    url('area_plot',views.area_plot,name='area_plot'),
    url('optimise',views.optimise,name='optimise'),
    url('carry',views.carry,name='carryover'),
]


urlpatterns += staticfiles_urlpatterns()

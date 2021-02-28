from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth.models import User,auth
from webdev.models import Destination
from webdev.models import Shop,UserCart,Reviews
from django.shortcuts import get_object_or_404
from django.contrib.auth.models import User
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth import login, authenticate, logout
from .housePricePrediction import Model
from .salaryprediction import Salary
from .sentimentanalyser import Sentiment
from .socialad import Socialad 
from .spam import Spam
from .handwrittendigit import Digit
from django.core.files.storage import FileSystemStorage
import re
import matplotlib.pyplot as plt 
from io import StringIO
from .hindiCharacterRecognition import HindiCharacterRecognition
from PIL import Image
import re
import base64
import io
import numpy as np
import skimage.io
import cv2 as cv






def home(request):
    allobjects = dict(Destination.objects.all())

    return render(request,'index.html',allobjects)

def register(request):
    #post=Destination()
    firstname=request.POST['fname']
    firstname=firstname.capitalize()
    lastname=request.POST['lname']
    username=request.POST['username']
    email=request.POST['email']
    pwd=request.POST['psw']
    user=User.objects.create_user(first_name=firstname,last_name=lastname,username=username,password=pwd,email=email)
    user.save()
    print("user created!!")
    
    return render(request,'index.html')

def login(request):
    
    if request.method=='POST':
        urname=request.POST.get('uname')
        urpwd=request.POST.get('psw')
        
        user = auth.authenticate(username=urname, password=urpwd)

        if user is not None:
            auth.login(request,user)
            return render(request,'index.html')
        else:
            return render(request,'invaliduser.html')

def logout(request):
    auth.logout(request)
    return render(request,'index.html')    

def aboutus(request):
    return render(request,'about.html')    

def regression(request):
    return render(request,'regression.html')    

def shop(request):

    products=list(Shop.objects.all())
    params={'products':products}
    print(params)
    return render(request,"shop.html",params)    

def add_to_cart(request):
    context={}
    flag='0'
    if request.user.is_authenticated:
        if request.method=="POST":
            print("HI")
            pnameid = request.POST["pnameid"]
            print(pnameid)
            is_exist = UserCart.objects.filter(productname_id=pnameid,username_id=request.user.id,status=False)
            if len(is_exist)>0:
                '''context["msz"] = "Item Already Exists in Your UserCart"
                context["cls"] = "alert alert-warning"'''
                pass
            else:   
                flag=0 
                '''product =get_object_or_404(Shop,id=pnameid)
                usr = get_object_or_404(User,id=request.user.id)'''
                c = UserCart(username_id=request.user.id,productname_id=pnameid,quantity=1,status=False)
                c.save()
                '''context["msz"] = "{} Added in Your Cart".format(product.productname_id)
                context["cls"] = "alert alert-success"'''
    else:
        print("youeedes")
        flag='1'

    items = UserCart.objects.filter(username_id=request.user.id,status=False)
    context["items"] = items
    context["flag"] = flag
    pricesum=0
    qtysum=0
    for i in items:
        pricesum+=(i.productname.productprice)*(i.quantity)
        qtysum+=i.quantity
    print(pricesum)    
    context["pricesum"]=pricesum
    context["qtysum"]= qtysum 
    
    return render(request,"cart.html",context)

def removecartitem(request):
    pnameid = request.GET["pnameid"]
    removableitems=UserCart.objects.filter(username_id=request.user.id,productname_id=pnameid,status=False)
    removableitems.delete()
    
    context={}
    
    items = UserCart.objects.filter(username_id=request.user.id,status=False)
    context["items"] = items
    pricesum=0
    qtysum=0
    for i in items:
        pricesum+=(i.productname.productprice)*(i.quantity)
        qtysum+=i.quantity          
    print(pricesum)    
    context["pricesum"]=pricesum
    context["qtysum"]= qtysum 

    return render(request,"cart.html",context)

def changequan(request):
    updatedqty=request.GET["quantityval"]
    productid=request.GET['pid']
    
    print(updatedqty,productid)
    
    record=UserCart.objects.get(username_id=request.user.id,productname_id=productid,status=False)
    pri=record.productname.productprice
    
    record.quantity=updatedqty
    record.save()   
    items = UserCart.objects.filter(username_id=request.user.id,status=False)
    context={}
    context["items"] = items   

    
    pricesum=0
    qtysum=0
    print(pri)
    qtyprodsum=int(pri)*int(updatedqty)
    print(qtyprodsum)
    for i in items:
        pricesum+=(i.productname.productprice)*(i.quantity)
        qtysum+=i.quantity 
    context["pricesum"]=pricesum
    context["qtysum"]= qtysum 
    return render(request,"cart.html",context)

def HousePricePrediction(request):
    return render(request,'HousePricePrediction.html',{"range":"House Price Predictor"})   

def HousePricePredictionResult(request):
    l1 = request.GET.getlist('input')
    num = int(l1[0])
    obj = Model()
    res = obj.predictor(num)
    print(num)
    print(res)
    return render(request,'HousePricePredictionResult.html',{"range":"House Price Predictor","input":str(num),"answer":str(res)})  

def SalaryPrediction(request):
     return render(request,"SalaryPrediction.html")   
def SalaryPredictionResult(request):
    exp=request.GET["yoe"]
    num=int(exp)
    obj=Salary()
    res=obj.predictor(num)
    print(res,num)
    
    return render(request,'salarypredictionresult.html',{"input":str(num),"answer":str(res)})

def sentimentanalyser(request):
    return render(request,'sentimentanalyser.html')
def sentimentanalyserresult(request):
      queryinput=str(request.GET["queryinput"])
      print(queryinput)
      obj=Sentiment()
      res=obj.predictor(queryinput)
      if res==1:
          ans=True
      else:
          ans=False    
      return render(request,'sentimentanalyserresult.html',{"ans":ans,"input":queryinput})

def digitrecog(request):
     return render(request,'digitrecog.html')

def digitrecogresult(request):
    print("HELLO ********************************")
    myfile=request.FILES['myfile']
    print(myfile)

    fs=FileSystemStorage()
    filename = fs.save(myfile.name, myfile)
    uploaded_file_url = fs.url(filename)

    print(uploaded_file_url)
    obj=Digit()
    res=obj.prediction(myfile.name)

    print(res)
    return render(request,'digitrecogresult.html',{"ans":res})

def addreview(request):
      print("hello")
      firstname=request.GET["firstname"]
      print("firstname")
      lastname=request.GET["lastname"] 
      Country=request.GET["country"]
      Subject=request.GET["subject"]

      c = Reviews(fname=firstname,lname=lastname,Country=Country,Subject=Subject)
      c.save()
      return render(request,"about.html")
        
def classification(request):
    return render(request,'classification.html')      

def socialad(request):
    return render(request,'socialad.html')  

def socialadresult(request):
    age=int(request.GET["age"])
    print(age)
    sal=int(request.GET["sal"])
    print(sal)

    obj=Socialad()
    res=int(obj.predictor(age,sal))
    
    return render(request,'socialadresult.html',{"ans":res})

def catordog(request):
    if(request.method=='POST'):
        img=request.FILES['imgfile']
        fs=FileSystemStorage()
        imgname=fs.save(img.name,img)
        cod=CatorDog()
        prediction=cod.predict(img.name)

        return render(request,'catordog.html',{'prediction':prediction})

    else:
        return render(request,'catordog.html')   

def spam(request):
    return render(request,'spam.html')
def spamresult(request):
    msg=request.GET['msg']
    obj=Spam()
    res=int(obj.predictor(msg))
    print(res)
    return render(request,'spamresult.html',{'ans':res})


def canvas(request):
    return render(request,'canvas.html')   



def savecanvas(request):
    
    canvasData =request.POST.get('inp1')

    datauri = canvasData
    imgstr = re.search(r'base64,(.*)', datauri).group(1)
    base64_decoded = base64.b64decode(imgstr)
    img = skimage.io.imread(io.BytesIO(base64_decoded),as_gray=True)
    img = cv.resize(img, dsize=(28, 28))
    #print(img)
    obj=HindiCharacterRecognition()
    res =obj.predictor(img)
    #print(res)
    return render(request,'canvas.html',{"ans":res})

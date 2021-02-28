from django.db import models
from django.contrib.auth.models import User,auth
from django.conf import settings
from django.template import RequestContext
# Create your models here.

class Destination(models.Model):
    firstname=models.CharField(max_length=100)
    lastname=models.CharField(max_length=100)
    username=models.CharField(max_length=100)
    email=models.CharField(max_length=50)
    password=models.CharField(max_length=50)
    
class Shop(models.Model):
    
    productid=models.AutoField
    productname=models.CharField(max_length=100)
    productdesc=models.CharField(max_length=1000)  
    productprice=models.IntegerField()
    productimage=models.ImageField(upload_to="shop/images")

    def __str__(self):
        return self.productname


User=settings.AUTH_USER_MODEL

class UserCart(models.Model):
    username=models.ForeignKey(User,on_delete=models.CASCADE)
    productname=models.ForeignKey(Shop,on_delete=models.CASCADE)
    quantity=models.IntegerField(default=1)
    status=models.BooleanField(default=False)

    def __str__(self):
        return str(self.username.username)

class Reviews(models.Model):
    fname=models.CharField(max_length=100)
    lname=models.CharField(max_length=100)
    Country=models.CharField(max_length=50)
    Subject=models.TextField(max_length=1000)    
    

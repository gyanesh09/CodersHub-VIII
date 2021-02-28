from django.contrib import admin
from .models import Destination
from .models import Shop,UserCart
from .models import Reviews

# Register your models here.

admin.site.register(Destination)
admin.site.register(Shop)
admin.site.register(UserCart)
admin.site.register(Reviews)

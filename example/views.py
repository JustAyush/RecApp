from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import json
from django.core import serializers
from django.template.loader import render_to_string

export = []
with open('exportNew.json', 'r') as myfile:
    export = json.load(myfile)

bookDetail = ''


# Create your views here.
def index(request):

    global export

    if request.user.is_authenticated:
        if (request.user.profile.role == 'Shopkeeper'):
            shopkeeper = 'yes'
        else: 
            shopkeeper = 'no'
    else: 
        shopkeeper = 'no'

    
    return render(request, 'example/index.html', {'e': export, 'shopkeeper': shopkeeper})


def detail(request, isbn):

    global export
    global bookDetail

    for book in export:
        if book['ISBN'] == isbn:
            bookDetail = book

    if request.user.is_authenticated:
        if (request.user.profile.role == 'Shopkeeper'):
            shopkeeper = 'yes'
        else: 
            shopkeeper = 'no'
    else: 
        shopkeeper = 'no'

    
    
    return render(request, 'example/detail.html', {'b': bookDetail, 'e': export, 'shopkeeper': shopkeeper})



# def index(request):
#     with open('export.json', 'r') as myfile:
#         export = json.load(myfile)
#         print(type(j))
#         return render(request, 'example/index.html', )
    
    

    



    
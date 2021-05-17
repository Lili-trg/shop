from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from django.http import HttpResponseRedirect, HttpResponse
from django.shortcuts import render


# Create your views here.
from vincenzo.forms import UserForm


def authorization(request):
    if request.method == "POST":
        form = UserForm(request.POST)
        if form.is_valid():
            username = request.POST['username']
            password = request.POST['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                # Потом изменить
                return HttpResponse('Залогинен!')
            else:
                error_message = "Неправильно введены данные"
                return render(request, 'authorization.html', {"error_message": error_message})
        else:
            error_message = "Некорректно введены данные"
            return render(request, 'authorization.html', {"error_message": error_message})

    return render(request, 'authorization.html')


def registration(request):
    if request.method == "POST":
        form = UserForm(request.POST)
        if form.is_valid():
            username = request.POST['username']
            password = request.POST['password']

            try:
                user = User.objects.get(username=username)
                error_message = "Пользователь с таким никнеймом уже зарегистрирован"
                return render(request, 'registration.html', {"error_message": error_message})
            except User.DoesNotExist:
                User.objects.create_user(username=username, password=password)

            return HttpResponseRedirect('..')
        else:
            error_message = "Некорректно введены данные"
            return render(request, 'registration.html', {"error_message": error_message})

    return render(request, 'registration.html')

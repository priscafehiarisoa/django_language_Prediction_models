from django.shortcuts import render

from django_language_test_models.codage.CheckLanguage import predict_language
from django_language_test_models.codage.CheckLanguage import checkIfCode
from django_language_test_models.form.LangageForm import LangageForm


def language_input(request):
    if request.method == 'POST':
        form = LangageForm(request.POST)
        if form.is_valid():
            language = form.cleaned_data['language']
            language=language.replace(" ","")
            language=language.replace("'","")
            language=language.replace("\"","")
            language=language.replace("[","")
            language=language.replace("]","")

            langage_list=language.split(",")
            predicted = predict_language(langage_list)
            sardinas= checkIfCode(langage_list)

            print(langage_list,"predicted 2:",predict_language(langage_list),"sardinas:",checkIfCode(langage_list))
            return render(request, './result.html', {'language': langage_list,"predicted":predicted,"sardinas":sardinas})
    else:
        form = LangageForm()
    return render(request, './input.html', {'form': form})
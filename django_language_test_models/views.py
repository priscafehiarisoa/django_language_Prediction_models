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
            text_predicted= "text-success" if predicted else "text-danger"
            sardinas= checkIfCode(langage_list)
            text_sardinas = "text-success" if sardinas else "text-danger"

            print(langage_list,"predicted 2:",predict_language(langage_list),"sardinas:",checkIfCode(langage_list))
            return render(request, './result.html', {'language': langage_list,"predicted":predicted,"sardinas":sardinas, "text_sardinas":text_sardinas, "text_prediction":text_predicted})
    else:
        form = LangageForm()
    return render(request, './input.html', {'form': form})


# EXEMPLES
#
# <form method="post" id="form1">
# {% csrf_token %}
# {{ form1.as_p }}
# <button type="submit" class="btn btn-primary">Submit Form 1</button>
# </form>
#
# <form method="post" id="form2">
# {% csrf_token %}
# {{ form2.as_p }}
# <button type="submit" class="btn btn-secondary">Submit Form 2</button>
# </form>
#
# def language_input(request):
#     if request.method == 'POST':
#         form1 = LangageForm(request.POST, prefix='form1')
#         form2 = AnotherForm(request.POST, prefix='form2')
#
#         if 'form1' in request.POST and form1.is_valid():
#             language = form1.cleaned_data['language']
#             language = language.replace(" ", "").replace("'", "").replace("\"", "").replace("[", "").replace("]", "")
#             langage_list = language.split(",")
#             predicted = predict_language(langage_list)
#             text_predicted = "text-success" if predicted else "text-danger"
#             sardinas = checkIfCode(langage_list)
#             text_sardinas = "text-success" if sardinas else "text-danger"
#             return render(request, './result.html', {'language': langage_list, "predicted": predicted, "sardinas": sardinas, "text_sardinas": text_sardinas, "text_prediction": text_predicted})
#
#         elif 'form2' in request.POST and form2.is_valid():
#             # Handle the second form submission
#             # Add your logic here for form2
#             pass
#
#     else:
#         form1 = LangageForm(prefix='form1')
#         form2 = AnotherForm(prefix='form2')
#
#     return render(request, './input.html', {'form1': form1, 'form2': form2})





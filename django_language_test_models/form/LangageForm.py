from django import forms


class LangageForm(forms.Form):
    language = forms.CharField(label='Enter language', max_length=1000, widget=forms.Textarea(attrs={'class': "form-control bg-transparent text-muted"}))

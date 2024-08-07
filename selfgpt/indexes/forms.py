from django import forms

from selfgpt.indexes.models import Index, IndexFile


class IndexForm(forms.ModelForm):
    class Meta:
        model = Index
        fields = ["name"]
        widgets = {
            "name": forms.TextInput(attrs={"class": "form-control w-50"}),
        }


class IndexFileForm(forms.ModelForm):
    class Meta:
        model = IndexFile
        fields = ["file"]
        widgets = {
            "file": forms.ClearableFileInput(attrs={"class": "form-control"}),
        }


class UserInputForm(forms.Form):
    user_input = forms.CharField(label="Your Message", max_length=1000)


class VideoURLForm(forms.Form):
    video_url = forms.URLField(
        label="YouTube Video URL", max_length=1000, widget=forms.URLInput(attrs={"class": "form-control"})
    )

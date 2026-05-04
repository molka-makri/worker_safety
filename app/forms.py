from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

ROLE_CHOICES = [
    ('member', 'Membre'),
    ('admin', 'Administrateur'),
    ('viewer', 'Observateur'),
    ('operator', 'Opérateur'),
]

class SignUpForm(UserCreationForm):
    email = forms.EmailField(required=True, widget=forms.EmailInput(attrs={'autocomplete': 'email'}))
    first_name = forms.CharField(required=False, max_length=30, widget=forms.TextInput(attrs={'autocomplete': 'given-name'}))
    last_name = forms.CharField(required=False, max_length=30, widget=forms.TextInput(attrs={'autocomplete': 'family-name'}))
    role = forms.ChoiceField(choices=ROLE_CHOICES, widget=forms.RadioSelect, initial='member')
    terms = forms.BooleanField(required=True, label="J'accepte les conditions d'utilisation")

    class Meta:
        model = User
        fields = ('username', 'email', 'first_name', 'last_name', 'password1', 'password2', 'role', 'terms')

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.first_name = self.cleaned_data.get('first_name', '')
        user.last_name = self.cleaned_data.get('last_name', '')
        if commit:
            user.save()
        return user

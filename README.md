# Projet PI - Interface Django

Interface web moderne et responsive construite avec Django, HTML, CSS et JavaScript.

## 🚀 Démarrage Rapide

### Prérequis
- Python 3.8+
- pip

### Installation

1. **Cloner ou naviguer vers le dossier du projet**
```bash
cd projet_pi
```

2. **Créer un environnement virtuel**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Appliquer les migrations**
```bash
python manage.py migrate
```

5. **Créer un superutilisateur (optionnel)**
```bash
python manage.py createsuperuser
```

6. **Lancer le serveur de développement**
```bash
python manage.py runserver
```

7. **Accéder à l'application**
- Interface: http://localhost:8000
- Admin: http://localhost:8000/admin

## 📁 Structure du Projet

```
projet_pi/
├── manage.py                 # Utilitaire de gestion Django
├── requirements.txt          # Dépendances Python
├── config/                   # Configuration Django
│   ├── settings.py          # Paramètres principaux
│   ├── urls.py              # URLs principales
│   ├── wsgi.py              # Configuration WSGI
│   └── __init__.py
├── app/                      # Application principale
│   ├── models.py            # Modèles de données
│   ├── views.py             # Vues/Contrôleurs
│   ├── urls.py              # URLs de l'app
│   ├── admin.py             # Configuration admin
│   ├── apps.py
│   ├── migrations/          # Migrations BD
│   └── __init__.py
├── templates/               # Fichiers HTML
│   ├── base.html            # Template principal
│   └── index.html           # Page d'accueil
├── static/                  # Fichiers statiques
│   ├── css/
│   │   └── style.css        # Styles CSS
│   ├── js/
│   │   └── main.js          # Script JavaScript
│   ├── images/              # Images
│   └── fonts/               # Polices
└── media/                   # Fichiers utilisateur
```

## 🎨 Fonctionnalités

- ✅ Interface responsive et moderne
- ✅ Navigation fluide
- ✅ Système de cartes et grilles
- ✅ API test intégrée
- ✅ JavaScript interactif
- ✅ Animations smooth
- ✅ Support mobile

## 📝 Ajouter vos Modèles

Modifiez le fichier `app/models.py`:

```python
from django.db import models

class MonModele(models.Model):
    nom = models.CharField(max_length=100)
    description = models.TextField()
    date_creation = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.nom
```

Puis:
```bash
python manage.py makemigrations
python manage.py migrate
```

## 📱 JavaScript Utiles

L'application inclut des fonctions JavaScript pratiques:

```javascript
// Requêtes API
await apiGet('/api/endpoint');
await apiPost('/api/endpoint', {data: 'valeur'});
await apiPut('/api/endpoint', {data: 'nouvelle_valeur'});
await apiDelete('/api/endpoint');

// Notifications
showNotification('Message', 'success');
showNotification('Erreur', 'error');

// Validation JSON
validateJSON(str); // true/false
```

## 🔧 Configuration

### Base de Données
Le projet utilise SQLite par défaut. Pour utiliser PostgreSQL:

```python
# config/settings.py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'projet_pi',
        'USER': 'username',
        'PASSWORD': 'password',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}
```

### Variables d'Environnement
Créez un fichier `.env`:

```
SECRET_KEY=your-secret-key
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
```

## 📚 Ressources

- [Django Documentation](https://docs.djangoproject.com/)
- [MDN Web Docs](https://developer.mozilla.org/)
- [Font Awesome Icons](https://fontawesome.com/)

## 🤝 Contribution

N'hésitez pas à améliorer et ajouter des fonctionnalités!

## 📄 Licence

Ce projet est sous licence libre.

---

**Prêt à commencer? Lancez le serveur et explorez l'interface!** 🎉

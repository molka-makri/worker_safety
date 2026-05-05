from django.test import TestCase, Client
from django.urls import reverse


class IndexViewTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.index_url = reverse('app:index')

    def test_index_view_status_code(self):
        """Tester que la page d'accueil retourne 200"""
        response = self.client.get(self.index_url)
        self.assertEqual(response.status_code, 200)

    def test_index_view_template(self):
        """Tester que le bon template est utilisé"""
        response = self.client.get(self.index_url)
        self.assertTemplateUsed(response, 'index.html')

    def test_index_view_context(self):
        """Tester le contexte de la vue"""
        response = self.client.get(self.index_url)
        self.assertIn('app_name', response.context)


class APITestViewTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.api_url = reverse('app:api_test')

    def test_api_post_request(self):
        """Tester une requête POST sur l'API"""
        response = self.client.post(
            self.api_url,
            data='{"test": "data"}',
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)

    def test_api_get_request_not_allowed(self):
        """Tester qu'une requête GET retourne une erreur"""
        response = self.client.get(self.api_url)
        self.assertEqual(response.status_code, 405)

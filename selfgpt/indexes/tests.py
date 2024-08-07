from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase

from .models import Index, IndexFile

User = get_user_model()


class IndexModelTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        # Set up non-modified objects used by all test methods
        cls.user = User.objects.create_user(email="john@example.com", password="12345")
        cls.index = Index.objects.create(name="Test Index", user=cls.user)

    def test_index_creation(self):
        self.assertEqual(self.index.name, "Test Index")
        self.assertEqual(self.index.user, self.user)

    def test_index_str(self):
        self.assertEqual(str(self.index), f"name: {self.index.name} / UUID: {self.index.uuid}")


class IndexFileModelTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = User.objects.create_user(email="john@example.com", password="12345")
        cls.index = Index.objects.create(name="Test Index", user=cls.user)

        # Create a mock file for testing
        mock_file = SimpleUploadedFile("test_file.txt", b"file_content", content_type="text/plain")

        cls.index_file = IndexFile.objects.create(index=cls.index, file=mock_file, original_filename="test_file.txt")

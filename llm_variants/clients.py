"""Clients LLM interchangeables : OpenAI, Hugging Face, et un faux client.

Chaque client expose ``complete(prompt) -> str``. Les clés d'API sont lues
dans les variables d'environnement (``OPENAI_API_KEY``, ``HF_TOKEN``) et ne
sont jamais écrites dans le code.
"""

import os
import time


class OpenAIClient:
    """Client pour l'API OpenAI (Chat Completions)."""

    def __init__(self, model="gpt-4o-mini", temperature=0.1, max_tokens=500):
        """Crée le client ; nécessite la variable OPENAI_API_KEY."""
        from openai import OpenAI

        self.client = OpenAI(timeout=60, max_retries=3)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, prompt):
        """Envoie un échange à tour unique et renvoie le texte produit."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""


class HuggingFaceClient:
    """Client pour l'API Hugging Face Inference (modèles ouverts)."""

    def __init__(self, model="Qwen/Qwen2.5-7B-Instruct", temperature=0.1,
                 max_tokens=500):
        """Crée le client ; utilise la variable HF_TOKEN si elle existe."""
        from huggingface_hub import InferenceClient

        self.client = InferenceClient(model=model,
                                      token=os.environ.get("HF_TOKEN"))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, prompt):
        """Envoie un échange à tour unique et renvoie le texte produit."""
        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""


class FakeClient:
    """Faux client déterministe pour les tests et les essais à sec.

    ``answers`` associe un rsID à une réponse ; les autres variants reçoivent
    ``default``. Aucun appel réseau n'est effectué.
    """

    model = "fake"

    def __init__(self, answers=None, default="Information non disponible"):
        """Crée le faux client."""
        self.answers = answers or {}
        self.default = default
        self.calls = []

    def complete(self, prompt):
        """Renvoie la réponse prévue pour le premier rsID trouvé dans le prompt."""
        self.calls.append(prompt)
        for variant, answer in self.answers.items():
            if variant in prompt:
                return answer
        return self.default


def make_client(provider, model=None, temperature=0.1):
    """Crée le client correspondant au fournisseur demandé."""
    if provider == "openai":
        return OpenAIClient(model or "gpt-4o-mini", temperature)
    if provider == "hf":
        return HuggingFaceClient(model or "Qwen/Qwen2.5-7B-Instruct",
                                 temperature)
    if provider == "dry-run":
        return FakeClient()
    raise ValueError(f"Fournisseur inconnu : {provider}")


def complete_with_delay(client, prompt, delay):
    """Appelle le client puis attend ``delay`` secondes (limites de débit)."""
    answer = client.complete(prompt)
    if delay:
        time.sleep(delay)
    return answer

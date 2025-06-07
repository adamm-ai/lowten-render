#!/bin/bash

# Activer l'environnement virtuel si vous en utilisez un
# source venv/bin/activate

# Installer les dépendances si nécessaire
pip install -r requirements.txt

# Démarrer l'application
uvicorn app:app --reload --host 0.0.0.0 --port 8000


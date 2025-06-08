#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MUBAGPT - APPLICATION FASTAPI POUR RENDER
Version optimisée pour le déploiement
"""

import os
import sys
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Créer l'application FastAPI
app = FastAPI(
    title="MubaGPT",
    description="Assistant immobilier intelligent",
    version="1.0.0"
)

# Configuration CORS pour les requêtes cross-origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifiez vos domaines
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================================================================
# CONFIGURATION DES FICHIERS STATIQUES
# ========================================================================================

def setup_static_directories():
    """Crée les dossiers statiques s'ils n'existent pas"""
    static_dirs = [
        "static",
        "static/css", 
        "static/js", 
        "static/images"
    ]
    
    for dir_path in static_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"📁 Créé le dossier: {dir_path}")
        else:
            logger.info(f"✅ Dossier existant: {dir_path}")

# Configurer les dossiers
setup_static_directories()

# Monter les fichiers statiques
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    logger.info("✅ Fichiers statiques montés sur /static")
except Exception as e:
    logger.error(f"❌ Erreur montage fichiers statiques: {e}")

# ========================================================================================
# ROUTES PRINCIPALES
# ========================================================================================

@app.get("/")
async def serve_index():
    """Sert la page principale index.html"""
    try:
        # Chercher index.html dans static/ en premier
        if os.path.exists("static/index.html"):
            logger.info("📄 Serving static/index.html")
            return FileResponse("static/index.html", media_type="text/html")
        
        # Fallback vers la racine
        elif os.path.exists("index.html"):
            logger.info("📄 Serving index.html from root")
            return FileResponse("index.html", media_type="text/html")
        
        else:
            logger.error("❌ index.html non trouvé")
            return JSONResponse(
                status_code=404,
                content={
                    "error": "Page non trouvée",
                    "message": "index.html est introuvable",
                    "checked_paths": ["static/index.html", "index.html"]
                }
            )
    
    except Exception as e:
        logger.error(f"❌ Erreur serving index: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Erreur serveur", "message": str(e)}
        )

@app.get("/health")
async def health_check():
    """Route de vérification de santé"""
    return {
        "status": "healthy",
        "message": "MubaGPT API is running",
        "version": "1.0.0"
    }

# ========================================================================================
# ROUTES DE DEBUG (UTILES POUR RENDER)
# ========================================================================================

@app.get("/debug/files")
async def debug_files():
    """Debug: Vérifier quels fichiers existent"""
    try:
        files_info = {}
        
        # Fichiers à vérifier
        check_files = [
            "static/index.html",
            "index.html", 
            "static/css/styles.css",
            "static/js/app.js",
            "app.py",
            "requirements.txt"
        ]
        
        for file_path in check_files:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                files_info[file_path] = {
                    "exists": True,
                    "size_bytes": size,
                    "size_kb": round(size / 1024, 2)
                }
            else:
                files_info[file_path] = {"exists": False}
        
        # Contenu du dossier static
        static_content = {}
        if os.path.exists("static"):
            for root, dirs, files in os.walk("static"):
                static_content[root] = {
                    "directories": dirs,
                    "files": files
                }
        
        # Contenu racine
        root_files = []
        try:
            root_files = [f for f in os.listdir(".") if os.path.isfile(f)]
        except:
            root_files = ["Erreur lecture racine"]
        
        return {
            "files_status": files_info,
            "static_directory_tree": static_content,
            "root_directory_files": root_files,
            "current_working_directory": os.getcwd(),
            "python_path": sys.path[0]
        }
    
    except Exception as e:
        return {"error": f"Erreur debug: {str(e)}"}

@app.get("/debug/routes")
async def debug_routes():
    """Debug: Lister toutes les routes disponibles"""
    routes_info = []
    
    for route in app.routes:
        route_info = {
            "path": getattr(route, 'path', 'N/A'),
            "name": getattr(route, 'name', 'N/A'),
            "methods": list(getattr(route, 'methods', [])) if hasattr(route, 'methods') else []
        }
        routes_info.append(route_info)
    
    return {
        "total_routes": len(routes_info),
        "routes": routes_info
    }

@app.get("/debug/env")
async def debug_environment():
    """Debug: Variables d'environnement importantes"""
    env_vars = {}
    
    # Variables importantes à vérifier
    important_vars = [
        "PORT", "HOST", "RENDER", "RENDER_SERVICE_NAME",
        "PYTHONPATH", "PWD", "HOME"
    ]
    
    for var in important_vars:
        env_vars[var] = os.environ.get(var, "Non définie")
    
    return {
        "environment_variables": env_vars,
        "python_version": sys.version,
        "platform": sys.platform
    }

# ========================================================================================
# ROUTES FALLBACK POUR FICHIERS STATIQUES (SÉCURITÉ)
# ========================================================================================

@app.get("/css/styles.css")
async def serve_css_fallback():
    """Fallback pour servir le CSS directement"""
    css_path = "static/css/styles.css"
    if os.path.exists(css_path):
        return FileResponse(css_path, media_type="text/css")
    else:
        raise HTTPException(status_code=404, detail="CSS file not found")

@app.get("/js/app.js")
async def serve_js_fallback():
    """Fallback pour servir le JS directement"""
    js_path = "static/js/app.js"
    if os.path.exists(js_path):
        return FileResponse(js_path, media_type="application/javascript")
    else:
        raise HTTPException(status_code=404, detail="JS file not found")

# ========================================================================================
# GESTION DES ERREURS
# ========================================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Gestionnaire d'erreur 404 personnalisé"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Ressource non trouvée",
            "path": str(request.url.path),
            "message": "La ressource demandée n'existe pas",
            "available_routes": ["/", "/health", "/debug/files"]
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Gestionnaire d'erreur 500 personnalisé"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Erreur serveur interne",
            "message": "Une erreur inattendue s'est produite"
        }
    )

# ========================================================================================
# POINT D'ENTRÉE PRINCIPAL
# ========================================================================================

if __name__ == "__main__":
    # Configuration pour développement local
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"🚀 Démarrage MubaGPT sur {host}:{port}")
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=False,  # Désactivé en production
        log_level="info"
    )
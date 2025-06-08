#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MUBAGPT - CHATBOT IMMOBILIER ULTRA-INTELLIGENT
Version optimisée pour déploiement Render
Claude 4 Sonnet + Groq Llama3 + Intelligence Hybride
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import logging
import asyncio
import os
import re
from datetime import datetime, timedelta
from functools import lru_cache
import hashlib
import time
from dataclasses import dataclass
import uuid

# Importations avec gestion d'erreurs pour Render
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️ Groq not available - install with: pip install groq")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("⚠️ Anthropic not available - install with: pip install anthropic")

try:
    from sqlalchemy import create_engine
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    print("⚠️ SQLAlchemy not available - install with: pip install sqlalchemy psycopg2-binary")

try:
    from elasticsearch import Elasticsearch
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False
    print("⚠️ Elasticsearch not available - install with: pip install elasticsearch")

# Configuration des logs avec token tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========================================================================================
# CONFIGURATION RENDER-OPTIMISÉE
# ========================================================================================

# Configuration Base de données PostgreSQL (Render)
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "mubagpt")

# Configuration Elasticsearch Cloud (Render)
ES_INDEX = os.getenv("ES_INDEX", "properties")
ES_CLOUD_URL = os.getenv("ES_CLOUD_URL", "")
ES_API_KEY = os.getenv("ES_API_KEY", "")

# Configuration Groq (Render)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")

# Configuration Anthropic Claude (Render)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")

# Configuration du serveur (Render détecte automatiquement le port via env)
PORT = int(os.getenv("PORT", 10000))
HOST = os.getenv("HOST", "0.0.0.0")

# ========================================================================================
# INITIALISATION DES CONNEXIONS AVEC FALLBACKS
# ========================================================================================

logger.info("🔧 Initialisation des connexions...")

# Base de données PostgreSQL (optionnelle sur Render)
engine = None
if SQLALCHEMY_AVAILABLE and DB_PASSWORD:
    try:
        DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(DATABASE_URL)
        logger.info("✅ PostgreSQL connecté")
    except Exception as e:
        logger.warning(f"⚠️ PostgreSQL non disponible: {e}")

# Elasticsearch Cloud
es = None
if ELASTICSEARCH_AVAILABLE and ES_CLOUD_URL and ES_API_KEY:
    try:
        es = Elasticsearch(ES_CLOUD_URL, api_key=ES_API_KEY)
        if es.ping():
            logger.info("✅ Elasticsearch Cloud connecté")
        else:
            logger.warning("⚠️ Elasticsearch Cloud non disponible")
            es = None
    except Exception as e:
        logger.warning(f"⚠️ Elasticsearch erreur: {e}")
        es = None

# Clients IA
groq_client = None
anthropic_client = None

if GROQ_AVAILABLE and GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("✅ Groq client initialisé")
    except Exception as e:
        logger.warning(f"⚠️ Groq erreur: {e}")

if ANTHROPIC_AVAILABLE and ANTHROPIC_API_KEY:
    try:
        anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        logger.info("✅ Anthropic client initialisé")
    except Exception as e:
        logger.warning(f"⚠️ Anthropic erreur: {e}")

# ========================================================================================
# SYSTÈME DE TRACKING DES TOKENS AVANCÉ
# ========================================================================================

@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_estimate: float
    model: str
    timestamp: datetime
    usage_type: str  # 'search', 'conversation', 'intent_analysis'

class AdvancedTokenTracker:
    """Système de tracking des tokens pour les deux APIs"""
    
    def __init__(self):
        self.daily_usage = []
        self.total_usage = {
            'total_tokens': 0,
            'total_queries': 0,
            'total_cost': 0.0,
            'search_tokens': 0,
            'conversation_tokens': 0,
            'intent_analysis_tokens': 0
        }
        
        # Prix par 1K tokens
        self.pricing = {
            'llama3-70b-8192': {'input': 0.001, 'output': 0.001},
            'claude-3-5-sonnet-20241022': {'input': 0.003, 'output': 0.015},
            'mixtral-8x7b-32768': {'input': 0.001, 'output': 0.001}
        }
    
    def log_usage(self, usage: TokenUsage):
        """Log l'utilisation des tokens avec détails avancés"""
        self.daily_usage.append(usage)
        self.total_usage['total_tokens'] += usage.total_tokens
        self.total_usage['total_queries'] += 1
        self.total_usage['total_cost'] += usage.cost_estimate
        
        if usage.usage_type == 'search':
            self.total_usage['search_tokens'] += usage.total_tokens
        elif usage.usage_type == 'intent_analysis':
            self.total_usage['intent_analysis_tokens'] += usage.total_tokens
        else:
            self.total_usage['conversation_tokens'] += usage.total_tokens
        
        # Log détaillé dans le terminal
        logger.info("=" * 70)
        logger.info(f"🔍 TOKEN USAGE TRACKING - {usage.usage_type.upper()}")
        logger.info("=" * 70)
        logger.info(f"🤖 Model: {usage.model}")
        logger.info(f"📝 Prompt tokens: {usage.prompt_tokens:,}")
        logger.info(f"💬 Completion tokens: {usage.completion_tokens:,}")
        logger.info(f"🔢 Total tokens: {usage.total_tokens:,}")
        logger.info(f"💰 Cost estimate: ${usage.cost_estimate:.6f}")
        logger.info(f"📊 Type: {usage.usage_type}")
        logger.info(f"⏰ Timestamp: {usage.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 70)
    
    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calcule le coût estimé"""
        if model not in self.pricing:
            return 0.0
        
        pricing = self.pricing[model]
        input_cost = (prompt_tokens / 1000) * pricing['input']
        output_cost = (completion_tokens / 1000) * pricing['output']
        return input_cost + output_cost

# Instance globale du tracker
token_tracker = AdvancedTokenTracker()

# ========================================================================================
# BASE DE DONNÉES DE PROPRIÉTÉS DE FALLBACK
# ========================================================================================

SAMPLE_PROPERTIES = [
    {
        "id": "1",
        "city": "Casablanca",
        "neighborhood": "Casablanca Finance City",
        "property_type": "Appartement",
        "price": 3340000,
        "area": 158,
        "bedrooms": 3,
        "bathrooms": 2,
        "description": "Magnifique appartement moderne avec vue sur la ville",
        "features": "Vue mer, Parking, Ascenseur",
        "score": 9.2
    },
    {
        "id": "2", 
        "city": "Rabat",
        "neighborhood": "Hay Riad",
        "property_type": "Villa",
        "price": 2500000,
        "area": 200,
        "bedrooms": 4,
        "bathrooms": 3,
        "description": "Villa familiale spacieuse avec jardin",
        "features": "Jardin, Garage, Piscine",
        "score": 8.8
    },
    {
        "id": "3",
        "city": "Marrakech", 
        "neighborhood": "Guéliz",
        "property_type": "Studio",
        "price": 850000,
        "area": 60,
        "bedrooms": 1,
        "bathrooms": 1,
        "description": "Studio moderne parfait pour investissement",
        "features": "Meublé, Centre-ville, Transport",
        "score": 8.1
    },
    {
        "id": "4",
        "city": "Casablanca",
        "neighborhood": "Anfa",
        "property_type": "Appartement", 
        "price": 4200000,
        "area": 180,
        "bedrooms": 4,
        "bathrooms": 3,
        "description": "Appartement de luxe dans quartier résidentiel",
        "features": "Standing, Sécurité, Parking",
        "score": 9.5
    },
    {
        "id": "5",
        "city": "Casablanca",
        "neighborhood": "Ain Diab",
        "property_type": "Villa",
        "price": 5800000,
        "area": 300,
        "bedrooms": 5,
        "bathrooms": 4, 
        "description": "Villa de prestige avec vue mer",
        "features": "Vue mer, Piscine, Jardin, Garage",
        "score": 9.8
    },
    {
        "id": "6",
        "city": "Rabat",
        "neighborhood": "Souissi",
        "property_type": "Appartement",
        "price": 1200000,
        "area": 90,
        "bedrooms": 2,
        "bathrooms": 2,
        "description": "Appartement lumineux dans quartier calme",
        "features": "Lumineux, Calme, Proche centres",
        "score": 8.3
    }
]

# ========================================================================================
# ANALYSEUR D'INTENT INTELLIGENT AVEC CLAUDE
# ========================================================================================

class IntelligentIntentAnalyzer:
    """Analyseur d'intent utilisant Claude pour des décisions ultra-intelligentes"""
    
    def __init__(self):
        self.client = anthropic_client
        self.model = CLAUDE_MODEL
    
    async def analyze_intent_and_decide(self, user_message: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Analyse intelligente avec Claude pour décider des actions à entreprendre"""
        
        # Fallback si Claude non disponible
        if not self.client:
            return self._fallback_intent_analysis(user_message)
        
        # Construire le contexte de conversation
        context = ""
        if conversation_history and len(conversation_history) > 0:
            context = "\n\nCONTEXTE DE CONVERSATION PRÉCÉDENTE:\n"
            for msg in conversation_history[-5:]:
                role = "Client" if msg["role"] == "user" else "Assistant"
                context += f"{role}: {msg['content']}\n"
        
        prompt = f"""Tu es un expert immobilier au Maroc. Analyse ce message client et décide de la stratégie optimale.

MESSAGE CLIENT: "{user_message}"{context}

Analyse et réponds avec un JSON exact dans ce format:

{{
    "should_search": true/false,
    "search_urgency": "immediate/progressive/none",
    "client_qualification_level": "cold/warm/hot/ready_to_visit",
    "detected_criteria": {{
        "has_location": true/false,
        "has_budget": true/false,
        "has_property_type": true/false,
        "has_size_requirements": true/false,
        "timeline_mentioned": true/false
    }},
    "conversation_strategy": "search_and_present/qualify_more/provide_consultation/book_visit",
    "next_qualifying_questions": ["question1", "question2"],
    "commercial_intent": "information_seeking/serious_buyer/just_browsing/ready_to_buy"
}}

DÉCISIONS DE RECHERCHE:
- should_search = true si: critères spécifiques mentionnés OU client semble prêt
- should_search = false si: question générale, salutation, demande de conseil sans critères

JSON uniquement:"""

        try:
            start_time = time.time()
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=600,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            end_time = time.time()
            
            # Tracking des tokens
            usage = response.usage
            cost = token_tracker.calculate_cost(self.model, usage.input_tokens, usage.output_tokens)
            
            token_usage = TokenUsage(
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                total_tokens=usage.input_tokens + usage.output_tokens,
                cost_estimate=cost,
                model=self.model,
                timestamp=datetime.now(),
                usage_type="intent_analysis"
            )
            
            token_tracker.log_usage(token_usage)
            
            logger.info(f"🧠 Claude intent analysis time: {(end_time - start_time):.2f}s")
            
            # Parse de la réponse
            response_text = response.content[0].text.strip()
            
            # Nettoyage du JSON
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            try:
                intent_analysis = json.loads(response_text)
                logger.info(f"🧠 ANALYSE D'INTENT CLAUDE: {intent_analysis}")
                return intent_analysis
            except json.JSONDecodeError as e:
                logger.error(f"❌ Erreur parsing intent JSON: {str(e)}")
                return self._fallback_intent_analysis(user_message)
                
        except Exception as e:
            logger.error(f"❌ Erreur Claude intent analysis: {str(e)}")
            return self._fallback_intent_analysis(user_message)
    
    def _fallback_intent_analysis(self, user_message: str) -> Dict[str, Any]:
        """Analyse basique de fallback"""
        message_lower = user_message.lower()
        
        # Détecter si recherche nécessaire
        search_keywords = ['appartement', 'villa', 'maison', 'studio', 'cherche', 'veux', 'besoin']
        should_search = any(keyword in message_lower for keyword in search_keywords)
        
        return {
            "should_search": should_search,
            "search_urgency": "immediate" if should_search else "none",
            "client_qualification_level": "warm",
            "detected_criteria": {
                "has_location": any(loc in message_lower for loc in ['casablanca', 'rabat', 'marrakech']),
                "has_budget": any(word in message_lower for word in ['budget', 'prix', 'million']),
                "has_property_type": any(word in message_lower for word in search_keywords),
                "has_size_requirements": 'chambres' in message_lower or 'pièces' in message_lower,
                "timeline_mentioned": False
            },
            "conversation_strategy": "search_and_present" if should_search else "qualify_more",
            "next_qualifying_questions": [],
            "commercial_intent": "information_seeking"
        }

# ========================================================================================
# SYSTÈME DE GESTION DE CONVERSATION AVANCÉ
# ========================================================================================

class AdvancedConversationManager:
    """Gestionnaire de conversation ultra-intelligent avec qualification client"""
    
    def __init__(self):
        self.client = anthropic_client
        self.model = CLAUDE_MODEL
        self.conversations = {}  # session_id -> conversation_history
        self.client_profiles = {}  # session_id -> profil client
    
    def get_or_create_conversation(self, session_id: str) -> List[Dict]:
        """Récupère ou crée une nouvelle conversation"""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        return self.conversations[session_id]
    
    def get_or_create_client_profile(self, session_id: str) -> Dict:
        """Récupère ou crée un profil client"""
        if session_id not in self.client_profiles:
            self.client_profiles[session_id] = {
                "qualification_level": "cold",
                "budget_range": None,
                "preferred_locations": [],
                "property_type": None,
                "timeline": None,
                "contact_info": {},
                "visit_interest": False,
                "interaction_count": 0
            }
        return self.client_profiles[session_id]
    
    def update_client_profile(self, session_id: str, intent_analysis: Dict, search_results: Dict = None):
        """Met à jour le profil client basé sur l'analyse d'intent"""
        profile = self.get_or_create_client_profile(session_id)
        
        # Mise à jour du niveau de qualification
        profile["qualification_level"] = intent_analysis.get("client_qualification_level", "cold")
        profile["interaction_count"] += 1
        
        # Extraction des critères détectés
        if search_results and search_results.get("criteria"):
            criteria = search_results["criteria"]
            
            if criteria.get("price_min") or criteria.get("price_max"):
                profile["budget_range"] = {
                    "min": criteria.get("price_min"),
                    "max": criteria.get("price_max")
                }
            
            if criteria.get("locations"):
                profile["preferred_locations"].extend(criteria["locations"])
                profile["preferred_locations"] = list(set(profile["preferred_locations"]))
            
            if criteria.get("property_type"):
                profile["property_type"] = criteria["property_type"]
        
        # Détecter l'intérêt pour une visite
        if intent_analysis.get("conversation_strategy") == "book_visit":
            profile["visit_interest"] = True
        
        logger.info(f"👤 PROFIL CLIENT MIS À JOUR: {profile}")
    
    def add_message(self, session_id: str, role: str, content: str):
        """Ajoute un message à la conversation"""
        conversation = self.get_or_create_conversation(session_id)
        conversation.append({"role": role, "content": content})
        
        # Garder seulement les 15 derniers messages
        if len(conversation) > 15:
            self.conversations[session_id] = conversation[-15:]
    
    async def generate_intelligent_response(self, session_id: str, user_message: str, intent_analysis: Dict, search_results: Optional[Dict] = None) -> str:
        """Génère une réponse ultra-intelligente basée sur l'analyse d'intent"""
        
        # Fallback si Claude non disponible
        if not self.client:
            return self._fallback_response(user_message, search_results)
        
        # Mettre à jour le profil client
        self.update_client_profile(session_id, intent_analysis, search_results)
        
        # Ajouter le message utilisateur
        self.add_message(session_id, "user", user_message)
        conversation = self.get_or_create_conversation(session_id)
        client_profile = self.get_or_create_client_profile(session_id)
        
        # Construire le prompt système intelligent
        system_prompt = f"""Tu es un expert immobilier professionnel au Maroc. Ton objectif: qualifier le client et proposer des solutions.

PROFIL CLIENT ACTUEL:
- Niveau de qualification: {client_profile['qualification_level']}
- Interactions: {client_profile['interaction_count']}
- Budget: {client_profile['budget_range'] or 'Non défini'}
- Zones d'intérêt: {', '.join(client_profile['preferred_locations']) or 'Aucune'}
- Type recherché: {client_profile['property_type'] or 'Non défini'}
- Intérêt visite: {client_profile['visit_interest']}

ANALYSE D'INTENT ACTUELLE:
- Stratégie recommandée: {intent_analysis.get('conversation_strategy')}
- Niveau qualification: {intent_analysis.get('client_qualification_level')}
- Intent commercial: {intent_analysis.get('commercial_intent')}

RÈGLES:
- Langage naturel et professionnel
- Présente les résultats comme un portefeuille personnel
- Pose 1-2 questions stratégiques pour avancer
- Si client qualifié, propose une visite/contact"""

        # Ajouter les résultats de recherche si disponibles
        if search_results and search_results.get("results"):
            properties_context = f"\n\nRÉSULTATS DISPONIBLES:\n"
            properties_context += f"Tu as {len(search_results['results'])} propriétés correspondantes.\n\n"
            
            for i, prop in enumerate(search_results["results"][:3], 1):
                properties_context += f"Option {i}: {prop.get('property_type', 'Propriété')} à {prop.get('city', 'N/A')}\n"
                properties_context += f"   Prix: {prop.get('price', 'N/A')} DH - {prop.get('area', 'N/A')} m² - {prop.get('bedrooms', 'N/A')} ch\n"
                properties_context += f"   Quartier: {prop.get('neighborhood', 'N/A')}\n\n"
            
            system_prompt += properties_context

        try:
            start_time = time.time()
            
            # Préparer les messages pour Claude
            messages = []
            for msg in conversation:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Appel à Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1200,
                temperature=0.7,
                system=system_prompt,
                messages=messages
            )
            
            end_time = time.time()
            
            # Tracking des tokens
            usage = response.usage
            cost = token_tracker.calculate_cost(self.model, usage.input_tokens, usage.output_tokens)
            
            token_usage = TokenUsage(
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                total_tokens=usage.input_tokens + usage.output_tokens,
                cost_estimate=cost,
                model=self.model,
                timestamp=datetime.now(),
                usage_type="conversation"
            )
            
            token_tracker.log_usage(token_usage)
            
            logger.info(f"💬 Claude response time: {(end_time - start_time):.2f}s")
            
            # Extraire la réponse
            assistant_response = response.content[0].text
            
            # Ajouter la réponse à la conversation
            self.add_message(session_id, "assistant", assistant_response)
            
            return assistant_response
            
        except Exception as e:
            logger.error(f"❌ Erreur Claude conversation: {str(e)}")
            return self._fallback_response(user_message, search_results)
    
    def _fallback_response(self, user_message: str, search_results: Optional[Dict] = None) -> str:
        """Réponse de fallback quand Claude n'est pas disponible"""
        if search_results and search_results.get("results"):
            count = len(search_results["results"])
            return f"J'ai trouvé {count} propriété{'s' if count > 1 else ''} qui pourrai{'en' if count > 1 else ''}t vous intéresser. Voici les meilleures options selon vos critères."
        else:
            return "Je suis là pour vous aider dans votre recherche immobilière. Pouvez-vous me donner plus de détails sur ce que vous recherchez ?"

# ========================================================================================
# ANALYSEUR DE REQUÊTE OPTIMISÉ
# ========================================================================================

class OptimizedQueryAnalyzer:
    """Analyseur hybride : règles + LLM en fallback"""
    
    # Patterns pré-compilés pour performance
    PRICE_PATTERNS = [
        re.compile(r'entre\s+(\d+(?:\s*000)*)\s*(?:et|à|-)\s*(\d+(?:\s*000)*)', re.IGNORECASE),
        re.compile(r'de\s+(\d+(?:\s*000)*)\s*(?:à|-)\s*(\d+(?:\s*000)*)', re.IGNORECASE),
        re.compile(r'budget\s+(\d+(?:\s*000)*)\s*(?:à|-)\s*(\d+(?:\s*000)*)', re.IGNORECASE),
        re.compile(r'maximum\s+(\d+(?:\s*000)*)', re.IGNORECASE),
        re.compile(r'max\s+(\d+(?:\s*000)*)', re.IGNORECASE),
        re.compile(r'(\d+)\s*millions?', re.IGNORECASE),
        re.compile(r'(\d+),(\d+)\s*millions?', re.IGNORECASE),
    ]
    
    AREA_PATTERNS = [
        re.compile(r'(\d+)\s*(?:à|-)\s*(\d+)\s*m[²2]', re.IGNORECASE),
        re.compile(r'minimum\s+(\d+)\s*m[²2]', re.IGNORECASE),
    ]
    
    BEDROOM_PATTERNS = [
        re.compile(r'(\d+)\s*chambres?', re.IGNORECASE),
        re.compile(r'(\d+)\s*ch\b', re.IGNORECASE),
    ]
    
    KNOWN_LOCATIONS = {
        'casablanca', 'casa', 'rabat', 'marrakech', 'fes', 'tanger',
        'agdal', 'hay riad', 'souissi', 'centre ville', 'maarif',
        'ain diab', 'bourgogne', 'palmier', 'val fleuri', 'anfa', 'gueliz'
    }
    
    PROPERTY_TYPES = {
        'appartement', 'villa', 'maison', 'studio', 'duplex', 'triplex'
    }
    
    @classmethod
    def extract_with_rules(cls, query: str) -> Dict[str, Any]:
        """Extraction basée sur des règles (rapide, 0 token)"""
        text_lower = query.lower()
        criteria = {
            "price_min": None, "price_max": None,
            "area_min": None, "area_max": None,
            "bedrooms_min": None, "rooms_min": None,
            "locations": [], "property_type": None, "search_terms": []
        }
        
        # Prix
        for pattern in cls.PRICE_PATTERNS:
            match = pattern.search(text_lower)
            if match:
                if 'millions?' in pattern.pattern:
                    if len(match.groups()) == 2:  # X,Y millions
                        millions = float(match.group(1) + '.' + match.group(2))
                        criteria["price_max"] = int(millions * 1000000)
                    else:  # X millions
                        millions = int(match.group(1))
                        criteria["price_max"] = millions * 1000000
                elif len(match.groups()) == 2:
                    criteria["price_min"] = int(match.group(1).replace(' ', ''))
                    criteria["price_max"] = int(match.group(2).replace(' ', ''))
                elif len(match.groups()) == 1:
                    criteria["price_max"] = int(match.group(1).replace(' ', ''))
                break
        
        # Superficie
        for pattern in cls.AREA_PATTERNS:
            match = pattern.search(text_lower)
            if match:
                if len(match.groups()) == 2:
                    criteria["area_min"] = int(match.group(1))
                    criteria["area_max"] = int(match.group(2))
                elif len(match.groups()) == 1:
                    criteria["area_min"] = int(match.group(1))
                break
        
        # Chambres
        for pattern in cls.BEDROOM_PATTERNS:
            match = pattern.search(text_lower)
            if match:
                criteria["bedrooms_min"] = int(match.group(1))
                break
        
        # Locations
        for location in cls.KNOWN_LOCATIONS:
            if location in text_lower:
                criteria["locations"].append(location)
        
        # Type de propriété
        for prop_type in cls.PROPERTY_TYPES:
            if prop_type in text_lower:
                criteria["property_type"] = prop_type
                break
        
        return criteria

# ========================================================================================
# LLM GROQ LLAMA3 AVEC TRACKING
# ========================================================================================

class GroqLlama3Extractor:
    """Extracteur utilisant Groq Llama3 avec tracking des tokens"""
    
    def __init__(self):
        self.client = groq_client
        self.model = GROQ_MODEL
    
    def extract_criteria(self, query: str) -> Dict[str, Any]:
        """Extraction avec Groq Llama3 et tracking des tokens"""
        
        # Fallback si Groq non disponible
        if not self.client:
            return OptimizedQueryAnalyzer.extract_with_rules(query)
        
        prompt = f"""Analyse cette requête de recherche immobilière et extrait UNIQUEMENT les critères spécifiques mentionnés.
Requête: "{query}"

Réponds SEULEMENT avec un JSON valide dans ce format exact:
{{
    "price_min": null,
    "price_max": null,
    "area_min": null,
    "area_max": null,
    "bedrooms_min": null,
    "rooms_min": null,
    "locations": [],
    "property_type": null,
    "search_terms": []
}}

Règles importantes:
- Si un budget/prix est mentionné, extrait price_min et/ou price_max
- Si "X millions" mentionné, convertis en nombre (ex: 3 millions = 3000000)
- locations: quartiers/villes mentionnés
- property_type: appartement, villa, maison, studio, etc.
- search_terms: mots-clés descriptifs (piscine, garage, terrasse, etc.)

N'invente aucune valeur. Si non mentionné = null ou []

JSON:"""
        
        try:
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
                top_p=0.9
            )
            
            end_time = time.time()
            
            # Tracking des tokens
            usage = response.usage
            cost = token_tracker.calculate_cost(self.model, usage.prompt_tokens, usage.completion_tokens)
            
            token_usage = TokenUsage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                cost_estimate=cost,
                model=self.model,
                timestamp=datetime.now(),
                usage_type="search"
            )
            
            token_tracker.log_usage(token_usage)
            
            logger.info(f"⚡ Groq response time: {(end_time - start_time):.2f}s")
            
            response_text = response.choices[0].message.content.strip()
            
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            try:
                criteria = json.loads(response_text)
                return criteria
            except json.JSONDecodeError as e:
                logger.error(f"❌ Erreur de parsing JSON: {str(e)}")
                return OptimizedQueryAnalyzer.extract_with_rules(query)
            
        except Exception as e:
            logger.error(f"❌ Erreur Groq Llama3: {str(e)}")
            return OptimizedQueryAnalyzer.extract_with_rules(query)

# ========================================================================================
# SYSTÈME DE CACHE SIMPLE
# ========================================================================================

class SimpleCache:
    """Cache en mémoire simple pour éviter les appels répétés"""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, tuple] = {}
        self.max_size = max_size
    
    def _normalize_query(self, query: str) -> str:
        return re.sub(r'\s+', ' ', query.lower().strip())
    
    def get(self, query: str) -> Optional[Dict]:
        key = self._normalize_query(query)
        if key in self.cache:
            result, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(hours=1):
                logger.info(f"🎯 Cache HIT pour: {query[:50]}...")
                return result
            else:
                del self.cache[key]
        return None
    
    def set(self, query: str, result: Dict):
        key = self._normalize_query(query)
        
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[key] = (result, datetime.now())
        logger.info(f"💾 Cache MISS - Sauvegardé: {query[:50]}...")

# ========================================================================================
# ELASTICSEARCH QUERY BUILDER
# ========================================================================================

class ElasticsearchQueryBuilder:
    """Construit des requêtes Elasticsearch optimisées"""
    
    @staticmethod
    def build_query(criteria: Dict[str, Any], search_terms: str = "") -> Dict[str, Any]:
        query = {
            "bool": {
                "must": [],
                "filter": [],
            }
        }

        # Filtres numériques exacts
        if criteria.get("price_min") or criteria.get("price_max"):
            price_filter = {"range": {"price": {}}}
            if criteria.get("price_min"):
                price_filter["range"]["price"]["gte"] = criteria["price_min"]
            if criteria.get("price_max"):
                price_filter["range"]["price"]["lte"] = criteria["price_max"]
            query["bool"]["filter"].append(price_filter)

        if criteria.get("area_min") or criteria.get("area_max"):
            area_filter = {"range": {"area": {}}}
            if criteria.get("area_min"):
                area_filter["range"]["area"]["gte"] = criteria["area_min"]
            if criteria.get("area_max"):
                area_filter["range"]["area"]["lte"] = criteria["area_max"]
            query["bool"]["filter"].append(area_filter)

        if criteria.get("bedrooms_min"):
            query["bool"]["filter"].append({
                "range": {"bedrooms": {"gte": criteria["bedrooms_min"]}}
            })

        # Localisations
        if criteria.get("locations"):
            for location in criteria["locations"]:
                query["bool"]["must"].append({
                    "multi_match": {
                        "query": location,
                        "fields": ["city^2", "neighborhood^2"],
                        "fuzziness": "AUTO"
                    }
                })

        # Type de propriété
        if criteria.get("property_type"):
            query["bool"]["must"].append({
                "match": {"property_type": {"query": criteria["property_type"], "boost": 2}}
            })

        # search_terms
        if criteria.get("search_terms"):
            for term in criteria["search_terms"]:
                query["bool"]["must"].append({
                    "multi_match": {
                        "query": term,
                        "fields": ["description^1.5", "features^1.2"],
                        "fuzziness": "AUTO"
                    }
                })

        # Recherche textuelle générale (fallback)
        if not (criteria.get("locations") or criteria.get("property_type") or criteria.get("search_terms")) and search_terms:
            query["bool"]["must"].append({
                "multi_match": {
                    "query": search_terms,
                    "fields": ["description^1.2", "features^1.1"],
                    "fuzziness": "AUTO"
                }
            })

        # Si aucun must, on fait un match_all
        if not query["bool"]["must"]:
            query["bool"]["must"].append({"match_all": {}})

        return {
            "query": query,
            "sort": [
                {"_score": {"order": "desc"}},
                {"price": {"order": "asc"}}
            ],
            "size": 20
        }

# ========================================================================================
# FONCTION DE RECHERCHE AVEC FALLBACK
# ========================================================================================

def search_properties_with_fallback(criteria: Dict, query: str) -> List[Dict]:
    """Recherche avec fallback sur les données de sample"""
    results = []
    
    for prop in SAMPLE_PROPERTIES:
        score = 0
        
        # Filtrer par budget
        if criteria.get("price_max"):
            if prop["price"] <= criteria["price_max"]:
                score += 3
            elif prop["price"] <= criteria["price_max"] * 1.2:  # 20% de tolérance
                score += 1
            else:
                continue  # Exclure si trop cher
        
        # Filtrer par localisation
        if criteria.get("locations"):
            for location in criteria["locations"]:
                if location.lower() in prop["city"].lower() or location.lower() in prop["neighborhood"].lower():
                    score += 3
                    break
        
        # Filtrer par type
        if criteria.get("property_type"):
            if prop["property_type"].lower() == criteria["property_type"].lower():
                score += 3
        
        # Filtrer par chambres
        if criteria.get("bedrooms_min"):
            if prop["bedrooms"] >= criteria["bedrooms_min"]:
                score += 2
        
        # Recherche textuelle dans la description
        if query:
            query_words = query.lower().split()
            description_lower = prop["description"].lower()
            for word in query_words:
                if word in description_lower:
                    score += 0.5
        
        if score > 0:
            prop_copy = prop.copy()
            prop_copy["score"] = score
            results.append(prop_copy)
    
    # Trier par score
    results.sort(key=lambda x: x["score"], reverse=True)
    
    return results

# ========================================================================================
# INSTANCES GLOBALES
# ========================================================================================

cache = SimpleCache()
groq_extractor = GroqLlama3Extractor()
conversation_manager = AdvancedConversationManager()
intent_analyzer = IntelligentIntentAnalyzer()

# ========================================================================================
# FONCTION DE RECHERCHE PRINCIPALE
# ========================================================================================

def search_properties(query: str) -> Dict[str, Any]:
    """Recherche optimisée avec extraction Llama3 et fallback"""
    try:
        logger.info(f"🚀 DÉBUT RECHERCHE: '{query}'")
        
        # 1. Vérifier le cache
        cached_result = cache.get(query)
        if cached_result:
            logger.info(f"🎯 CACHE HIT")
            return cached_result
        
        # 2. Extraction via Groq/rules
        criteria = groq_extractor.extract_criteria(query)
        logger.info(f"📋 Critères extraits: {criteria}")
        
        # 3. Recherche Elasticsearch ou fallback
        results = []
        total = 0
        
        if es and ES_INDEX:
            try:
                # Recherche Elasticsearch
                es_query = ElasticsearchQueryBuilder.build_query(criteria, query)
                res = es.search(index=ES_INDEX, body=es_query)
                
                hits = res["hits"]["hits"]
                total = res["hits"]["total"]["value"] if isinstance(res["hits"]["total"], dict) else res["hits"]["total"]
                
                for hit in hits:
                    result = hit["_source"].copy()
                    result["score"] = hit["_score"]
                    if result.get("price") and result.get("area") and result["area"] > 0:
                        result["price_per_sqm"] = round(result["price"] / result["area"], 2)
                    results.append(result)
                
                logger.info(f"✅ Elasticsearch: {len(results)} résultats")
                
            except Exception as e:
                logger.warning(f"⚠️ Elasticsearch erreur, fallback: {e}")
                results = search_properties_with_fallback(criteria, query)
                total = len(results)
        else:
            # Fallback sur données de sample
            logger.info("📦 Utilisation des données de fallback")
            results = search_properties_with_fallback(criteria, query)
            total = len(results)
        
        # 4. Préparer le résultat final
        final_result = {
            "query": query,
            "criteria": criteria,
            "extraction_method": "GROQ_LLAMA3" if groq_client else "RULES",
            "total_results": total,
            "results": results,
            "data_source": "elasticsearch" if es else "fallback"
        }
        
        # 5. Sauvegarder en cache
        cache.set(query, final_result)
        
        logger.info(f"✅ RECHERCHE TERMINÉE: {len(results)} propriétés")
        return final_result
        
    except Exception as e:
        logger.error(f"❌ ERREUR RECHERCHE: {str(e)}")
        return {
            "query": query,
            "criteria": {},
            "extraction_method": "ERROR",
            "total_results": 0,
            "results": [],
            "error": str(e)
        }

# ========================================================================================
# ORCHESTRATEUR PRINCIPAL
# ========================================================================================

class UltraIntelligentOrchestrator:
    """Orchestrateur principal avec IA hybride Claude + Groq"""
    
    def __init__(self):
        self.conversation_manager = conversation_manager
        self.intent_analyzer = intent_analyzer
    
    async def process_message_intelligently(self, session_id: str, user_message: str) -> Dict[str, Any]:
        """Traite un message avec une intelligence artificielle hybride"""
        
        logger.info(f"🧠 DÉBUT PROCESSUS IA HYBRIDE - Session: {session_id}")
        
        # ÉTAPE 1: ANALYSE D'INTENT
        conversation_history = self.conversation_manager.get_or_create_conversation(session_id)
        
        try:
            intent_analysis = await self.intent_analyzer.analyze_intent_and_decide(
                user_message=user_message,
                conversation_history=conversation_history
            )
            logger.info(f"🎯 INTENT: {intent_analysis}")
        except Exception as e:
            logger.error(f"❌ Erreur analyse intent: {str(e)}")
            intent_analysis = {
                "should_search": True,
                "search_urgency": "immediate",
                "client_qualification_level": "warm",
                "conversation_strategy": "search_and_present"
            }
        
        # ÉTAPE 2: DÉCISION DE RECHERCHE
        search_results = None
        should_search = intent_analysis.get("should_search", False)
        search_urgency = intent_analysis.get("search_urgency", "none")
        
        if should_search and search_urgency in ["immediate", "progressive"]:
            logger.info(f"🔍 RECHERCHE - Urgence: {search_urgency}")
            try:
                search_results = search_properties(user_message)
                logger.info(f"✅ RECHERCHE: {search_results.get('total_results', 0)} résultats")
            except Exception as e:
                logger.error(f"❌ Erreur recherche: {str(e)}")
        
        # ÉTAPE 3: GÉNÉRATION DE RÉPONSE
        try:
            conversational_response = await self.conversation_manager.generate_intelligent_response(
                session_id=session_id,
                user_message=user_message,
                intent_analysis=intent_analysis,
                search_results=search_results
            )
        except Exception as e:
            logger.error(f"❌ Erreur génération réponse: {str(e)}")
            conversational_response = "Désolé, j'ai rencontré un problème technique. Pouvez-vous reformuler ?"
        
        # ÉTAPE 4: RÉPONSE FINALE
        client_profile = self.conversation_manager.get_or_create_client_profile(session_id)
        
        response = {
            "message": conversational_response,
            "sender": "bot",
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "intent_analysis": intent_analysis,
            "client_profile": client_profile,
            "had_search": should_search,
            "qualification_level": intent_analysis.get("client_qualification_level"),
            "commercial_intent": intent_analysis.get("commercial_intent"),
            "ready_for_visit": client_profile.get("visit_interest", False)
        }
        
        # Ajouter les résultats de recherche si disponibles
        if search_results and search_results.get("results"):
            top_results = search_results["results"][:3]
            other_results = search_results["results"][3:]
            
            response.update({
                "properties": top_results,
                "otherProperties": other_results,
                "totalResults": search_results["total_results"],
                "criteria": search_results["criteria"],
                "extractionMethod": search_results["extraction_method"]
            })
        
        logger.info(f"🧠 FIN PROCESSUS IA HYBRIDE - SUCCÈS")
        return response

# ========================================================================================
# API FASTAPI
# ========================================================================================

class QueryRequest(BaseModel):
    user_query: str
    session_id: Optional[str] = None

class MessageRequest(BaseModel):
    message: str
    sender: str
    session_id: Optional[str] = None

class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict):
        for connection in self.active_connections:
            await connection.send_json(message)

# Créer l'application FastAPI
app = FastAPI(title="MubaGPT - Chatbot Immobilier Ultra-Intelligent", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================================================================
# CONFIGURATION DES FICHIERS STATIQUES POUR RENDER
# ========================================================================================

def setup_static_directories():
    """Crée les dossiers statiques s'ils n'existent pas"""
    static_dirs = ["static", "static/css", "static/js", "static/images"]
    
    for dir_path in static_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"📁 Créé le dossier: {dir_path}")

setup_static_directories()

try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    logger.info("✅ Fichiers statiques montés sur /static")
except Exception as e:
    logger.warning(f"⚠️ Erreur montage fichiers statiques: {e}")

# ========================================================================================
# ROUTES ET WEBSOCKET
# ========================================================================================

# Instances globales des managers
manager = WebSocketManager()
orchestrator = UltraIntelligentOrchestrator()

@app.get("/")
async def serve_index():
    """Sert la page principale"""
    try:
        if os.path.exists("static/index.html"):
            return FileResponse("static/index.html", media_type="text/html")
        elif os.path.exists("index.html"):
            return FileResponse("index.html", media_type="text/html")
        else:
            return JSONResponse(
                status_code=404,
                content={"error": "Page non trouvée", "message": "index.html introuvable"}
            )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Erreur serveur"})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    session_id = str(uuid.uuid4())
    logger.info(f"🔌 Nouvelle connexion WebSocket: {session_id}")
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("sender") == "user":
                user_message = message_data.get("message", "")
                
                # Message de traitement
                await websocket.send_json({
                    "message": "L'assistant analyse votre demande...",
                    "sender": "bot",
                    "timestamp": datetime.now().isoformat(),
                    "session_id": session_id,
                    "is_typing": True
                })
                
                # Traitement intelligent
                response = await orchestrator.process_message_intelligently(session_id, user_message)
                
                # Envoi de la réponse finale
                await websocket.send_json(response)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info(f"🔌 Déconnexion WebSocket: {session_id}")
    except Exception as e:
        logger.error(f"❌ Erreur WebSocket: {e}")
        manager.disconnect(websocket)

@app.post("/chat")
async def chat_endpoint(request: MessageRequest):
    """Endpoint REST pour le chat conversationnel intelligent"""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        response = await orchestrator.process_message_intelligently(session_id, request.message)
        return response
    except Exception as e:
        logger.error(f"❌ Erreur chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur de chat: {str(e)}")

@app.post("/search")
async def search_endpoint(request: QueryRequest):
    """Endpoint REST pour la recherche pure"""
    try:
        results = search_properties(request.user_query)
        return results
    except Exception as e:
        logger.error(f"❌ Erreur recherche: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur de recherche: {str(e)}")

@app.get("/health")
def health_check():
    """Health check pour Render"""
    return {
        "status": "healthy",
        "services": {
            "elasticsearch": es is not None and (es.ping() if es else False),
            "groq": groq_client is not None,
            "claude": anthropic_client is not None,
            "database": engine is not None
        },
        "stats": {
            "total_queries": token_tracker.total_usage['total_queries'],
            "total_tokens": token_tracker.total_usage['total_tokens'],
            "active_conversations": len(conversation_manager.conversations),
            "active_websockets": len(manager.active_connections)
        },
        "version": "4.0-render"
    }

@app.get("/stats")
def get_stats():
    """Statistiques complètes"""
    try:
        daily_stats = {
            "queries": len(token_tracker.daily_usage),
            "tokens": sum(u.total_tokens for u in token_tracker.daily_usage),
            "cost": sum(u.cost_estimate for u in token_tracker.daily_usage),
        }
        
        qualification_stats = {
            "cold": len([p for p in conversation_manager.client_profiles.values() if p["qualification_level"] == "cold"]),
            "warm": len([p for p in conversation_manager.client_profiles.values() if p["qualification_level"] == "warm"]),
            "hot": len([p for p in conversation_manager.client_profiles.values() if p["qualification_level"] == "hot"]),
            "ready_to_visit": len([p for p in conversation_manager.client_profiles.values() if p["qualification_level"] == "ready_to_visit"]),
        }
        
        return {
            "daily_stats": daily_stats,
            "total_stats": token_tracker.total_usage,
            "qualification_stats": qualification_stats,
            "cache_stats": {"cache_size": len(cache.cache)},
            "conversation_stats": {
                "active_sessions": len(conversation_manager.conversations),
                "total_messages": sum(len(conv) for conv in conversation_manager.conversations.values()),
                "active_websockets": len(manager.active_connections)
            }
        }
    except Exception as e:
        return {"error": f"Erreur stats: {str(e)}"}

# Routes fallback pour compatibilité
@app.get("/css/styles.css")
async def serve_css_fallback():
    css_path = "static/css/styles.css"
    if os.path.exists(css_path):
        return FileResponse(css_path, media_type="text/css")
    raise HTTPException(status_code=404, detail="CSS file not found")

@app.get("/js/app.js")
async def serve_js_fallback():
    js_path = "static/js/app.js"
    if os.path.exists(js_path):
        return FileResponse(js_path, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="JS file not found")

# ========================================================================================
# INITIALISATION ET DÉMARRAGE
# ========================================================================================

def initialize_system():
    """Initialise le système avec vérifications"""
    try:
        logger.info("🔧 === INITIALISATION SYSTÈME RENDER ===")
        
        # Vérifier Elasticsearch
        if es:
            try:
                if es.ping():
                    logger.info("✅ Elasticsearch Cloud connecté")
                else:
                    logger.warning("⚠️ Elasticsearch Cloud non accessible")
            except:
                logger.warning("⚠️ Elasticsearch Cloud erreur de ping")
        else:
            logger.warning("⚠️ Elasticsearch non configuré - Utilisation des données de fallback")
        
        # Vérifier Groq
        if groq_client:
            logger.info("✅ Groq Llama3 configuré")
        else:
            logger.warning("⚠️ Groq non disponible - Extraction par règles activée")
        
        # Vérifier Claude
        if anthropic_client:
            logger.info("✅ Claude 4 Sonnet configuré")
        else:
            logger.warning("⚠️ Claude non disponible - Réponses de fallback activées")
        
        # Log configuration Render
        logger.info(f"🌐 Port: {PORT}")
        logger.info(f"🌐 Host: {HOST}")
        logger.info(f"📊 Sample properties: {len(SAMPLE_PROPERTIES)}")
        
        logger.info("🔧 === SYSTÈME INITIALISÉ POUR RENDER ===")
        
    except Exception as e:
        logger.error(f"❌ Erreur initialisation: {str(e)}")

# Initialiser le système
initialize_system()

# Point d'entrée principal pour Render
if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Démarrage MubaGPT sur Render")
    logger.info("🧠 Intelligence Hybride: Claude 4 Sonnet + Groq Llama3")
    logger.info("🎯 Système de qualification client automatique")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
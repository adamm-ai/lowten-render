


from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
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

# Importation Groq pour Llama3 (recherche)
from groq import Groq

# Importation Anthropic pour Claude 4 Sonnet (conversation)
import anthropic

# Configuration des logs avec token tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---

# Base de données PostgreSQL
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")

# Elasticsearch
ES_INDEX = os.getenv("ES_INDEX")
ES_CLOUD_URL = os.getenv("ES_CLOUD_URL")
ES_API_KEY = os.getenv("ES_API_KEY")

# Configuration Groq (recherche LLaMA3)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")

# Configuration Anthropic (Claude)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20240620")

# --- CONNEXIONS ---
from sqlalchemy import create_engine
from elasticsearch import Elasticsearch

logger.info("Initialisation des connexions...")
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
es = Elasticsearch(ES_CLOUD_URL, api_key=ES_API_KEY)

# Initialisation des clients IA
groq_client = Groq(api_key=GROQ_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# --- SYSTÈME DE TRACKING DES TOKENS AVANCÉ ---
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
            'claude-4-sonnet-20241022': {'input': 0.003, 'output': 0.015},
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

# --- ANALYSEUR D'INTENT INTELLIGENT AVEC CLAUDE ---
class IntelligentIntentAnalyzer:
    """Analyseur d'intent utilisant Claude 4 Sonnet pour des décisions ultra-intelligentes"""
    
    def __init__(self):
        self.client = anthropic_client
        self.model = CLAUDE_MODEL
    
    async def analyze_intent_and_decide(self, user_message: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Analyse intelligente avec Claude pour décider des actions à entreprendre"""
        
        # Construire le contexte de conversation
        context = ""
        if conversation_history and len(conversation_history) > 0:
            context = "\n\nCONTEXTE DE CONVERSATION PRÉCÉDENTE:\n"
            for msg in conversation_history[-5:]:  # Derniers 5 messages
                role = "Client" if msg["role"] == "user" else "Youssef"
                context += f"{role}: {msg['content']}\n"
        
        prompt = f"""Tu es Youssef, expert immobilier au Maroc. Analyse ce message client et décide de la stratégie optimale.

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

QUALIFICATION CLIENT:
- cold: Premier contact, pas de critères précis
- warm: Critères partiels, besoin de qualification 
- hot: Critères complets, prêt à voir des biens
- ready_to_visit: Exprime un intérêt pour visiter

STRATÉGIES:
- search_and_present: Lancer recherche et présenter les biens
- qualify_more: Poser des questions pour affiner
- provide_consultation: Donner des conseils d'expert
- book_visit: Proposer une visite/rendez-vous

JSON uniquement, pas d'explication:"""

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
                # Fallback basique
                return {
                    "should_search": True,
                    "search_urgency": "immediate",
                    "client_qualification_level": "warm",
                    "detected_criteria": {"has_location": False, "has_budget": False, "has_property_type": False},
                    "conversation_strategy": "search_and_present",
                    "next_qualifying_questions": [],
                    "commercial_intent": "information_seeking"
                }
                
        except Exception as e:
            logger.error(f"❌ Erreur Claude intent analysis: {str(e)}")
            # Fallback basique
            return {
                "should_search": True,
                "search_urgency": "immediate", 
                "client_qualification_level": "warm",
                "detected_criteria": {"has_location": False, "has_budget": False, "has_property_type": False},
                "conversation_strategy": "search_and_present",
                "next_qualifying_questions": [],
                "commercial_intent": "information_seeking"
            }

# --- SYSTÈME DE GESTION DE CONVERSATION AVANCÉ ---
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
                profile["preferred_locations"] = list(set(profile["preferred_locations"]))  # Unique
            
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
        
        # Mettre à jour le profil client
        self.update_client_profile(session_id, intent_analysis, search_results)
        
        # Ajouter le message utilisateur
        self.add_message(session_id, "user", user_message)
        conversation = self.get_or_create_conversation(session_id)
        client_profile = self.get_or_create_client_profile(session_id)
        
        # Construire le prompt système intelligent
        system_prompt = f"""Tu es Youssef, expert immobilier depuis 15 ans au Maroc. Ton objectif: qualifier le client et booker des visites.

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
- Questions suggérées: {intent_analysis.get('next_qualifying_questions', [])}

DIRECTIVES COMMERCIALES:
1. Si stratégie = "search_and_present": Présente les biens de manière engageante
2. Si stratégie = "qualify_more": Pose des questions intelligentes pour qualifier
3. Si stratégie = "provide_consultation": Donne des conseils d'expert 
4. Si stratégie = "book_visit": Propose activement une visite/rendez-vous

OBJECTIFS SELON QUALIFICATION:
- Cold (nouveau): Établir confiance, comprendre besoins
- Warm (intéressé): Présenter options, affiner critères
- Hot (prêt): Proposer les meilleures options, pousser vers visite
- Ready_to_visit: Booker immédiatement une visite

RÈGLES ABSOLUES:
- Pas d'emojis, langage naturel et professionnel
- Toujours dans la langue du client
- Présente les résultats comme ton portefeuille personnel
- Pose 1-2 questions stratégiques pour avancer la vente
- Si client qualifié, propose TOUJOURS une visite

STYLE: Expert confiant, commercial mais pas agressif, axé résultats."""

        # Ajouter les résultats de recherche si disponibles
        if search_results and search_results.get("results"):
            properties_context = f"\n\nRÉSULTATS DE RECHERCHE DISPONIBLES:\n"
            properties_context += f"Tu as {search_results.get('total_results', 0)} propriétés dans ton portefeuille qui correspondent.\n\n"
            
            if len(search_results["results"]) > 0:
                properties_context += "PROPRIÉTÉS À PRÉSENTER (choisir les meilleures selon le profil client):\n"
                for i, prop in enumerate(search_results["results"][:8], 1):
                    properties_context += f"\nOption {i}: {prop.get('property_type', 'Propriété')} à {prop.get('city', 'N/A')}\n"
                    properties_context += f"   Prix: {prop.get('price', 'À négocier')} DH\n"
                    properties_context += f"   Superficie: {prop.get('area', 'N/A')} m²\n"
                    properties_context += f"   Quartier: {prop.get('neighborhood', 'N/A')}\n"
                    properties_context += f"   Chambres: {prop.get('bedrooms', 'N/A')}\n"
                    properties_context += f"   Score: {prop.get('score', 0):.1f}/10\n"
                
                properties_context += f"\nSTRATÉGIE: Présente 2-3 meilleures options selon le profil, crée l'urgence, propose visite."
            
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
            
            # Appel à Claude 4 Sonnet avec prompt intelligent
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
            return "Désolé, j'ai rencontré un problème technique. Pouvez-vous reformuler votre demande ?"

# --- ANALYSEUR DE REQUÊTE OPTIMISÉ (CONSERVÉ) ---
class OptimizedQueryAnalyzer:
    """Analyseur hybride : règles + LLM en fallback"""
    
    # Patterns pré-compilés pour performance
    PRICE_PATTERNS = [
        re.compile(r'entre\s+(\d+(?:\s*000)*)\s*(?:et|à|-)\s*(\d+(?:\s*000)*)', re.IGNORECASE),
        re.compile(r'de\s+(\d+(?:\s*000)*)\s*(?:à|-)\s*(\d+(?:\s*000)*)', re.IGNORECASE),
        re.compile(r'budget\s+(\d+(?:\s*000)*)\s*(?:à|-)\s*(\d+(?:\s*000)*)', re.IGNORECASE),
        re.compile(r'maximum\s+(\d+(?:\s*000)*)', re.IGNORECASE),
        re.compile(r'max\s+(\d+(?:\s*000)*)', re.IGNORECASE),
        re.compile(r'moins\s+de\s+(\d+(?:\s*000)*)', re.IGNORECASE),
        re.compile(r'(\d+)\s*millions?', re.IGNORECASE),
        re.compile(r'(\d+),(\d+)\s*millions?', re.IGNORECASE),
    ]
    
    AREA_PATTERNS = [
        re.compile(r'(\d+)\s*(?:à|-)\s*(\d+)\s*m[²2]', re.IGNORECASE),
        re.compile(r'entre\s+(\d+)\s*(?:et|à|-)\s*(\d+)\s*m[²2]', re.IGNORECASE),
        re.compile(r'minimum\s+(\d+)\s*m[²2]', re.IGNORECASE),
    ]
    
    BEDROOM_PATTERNS = [
        re.compile(r'(\d+)\s*chambres?', re.IGNORECASE),
        re.compile(r'(\d+)\s*ch\b', re.IGNORECASE),
    ]
    
    ROOM_PATTERNS = [
        re.compile(r'(\d+)\s*pièces?', re.IGNORECASE),
        re.compile(r'(\d+)p\b', re.IGNORECASE),
    ]
    
    KNOWN_LOCATIONS = {
        'casablanca', 'casa', 'rabat', 'marrakech', 'fes', 'tanger',
        'agdal', 'hay riad', 'souissi', 'centre ville', 'maarif',
        'ain diab', 'bourgogne', 'palmier', 'val fleuri'
    }
    
    PROPERTY_TYPES = {
        'appartement', 'villa', 'maison', 'studio', 'duplex', 'triplex'
    }
    
    AMENITIES = {
        'piscine', 'garage', 'terrasse', 'jardin', 'ascenseur', 'parking'
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
                if pattern.pattern == r'(\d+)\s*millions?':
                    millions = int(match.group(1))
                    criteria["price_max"] = millions * 1000000
                elif pattern.pattern == r'(\d+),(\d+)\s*millions?':
                    millions = float(match.group(1) + '.' + match.group(2))
                    criteria["price_max"] = int(millions * 1000000)
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
        
        # Pièces
        for pattern in cls.ROOM_PATTERNS:
            match = pattern.search(text_lower)
            if match:
                criteria["rooms_min"] = int(match.group(1))
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
        
        # Amenities
        for amenity in cls.AMENITIES:
            if amenity in text_lower:
                criteria["search_terms"].append(amenity)
        
        return criteria

# --- LLM GROQ LLAMA3 AVEC TRACKING (CONSERVÉ) ---
class GroqLlama3Extractor:
    """Extracteur utilisant Groq Llama3 avec tracking des tokens"""
    
    def __init__(self):
        self.client = groq_client
        self.model = GROQ_MODEL
    
    def extract_criteria(self, query: str) -> Dict[str, Any]:
        """Extraction avec Groq Llama3 et tracking des tokens"""
        
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
- Si une superficie est mentionnée, extrait area_min et/ou area_max  
- Si nombre de chambres mentionné, extrait bedrooms_min
- Si nombre de pièces mentionné, extrait rooms_min
- locations: quartiers/villes mentionnés
- property_type: appartement, villa, maison, studio, etc.
- search_terms: mots-clés descriptifs (piscine, garage, terrasse, etc.)
- Si on mentionne "X millions" ou "X millions de dirhams/dh", convertis en nombre (ex: 3 millions = 3000000)

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

# --- SYSTÈME DE CACHE SIMPLE (CONSERVÉ) ---
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

# --- ELASTICSEARCH QUERY BUILDER (CONSERVÉ) ---
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

        if criteria.get("rooms_min"):
            query["bool"]["filter"].append({
                "range": {"rooms": {"gte": criteria["rooms_min"]}}
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
                        "fields": ["description^1.5", "features^1.2", "extras^1.1"],
                        "fuzziness": "AUTO"
                    }
                })

        # Recherche textuelle générale (fallback)
        if not (criteria.get("locations") or criteria.get("property_type") or criteria.get("search_terms")) and search_terms:
            query["bool"]["must"].append({
                "multi_match": {
                    "query": search_terms,
                    "fields": ["full_text^1", "description^1.2", "features^1.1"],
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

# Instances globales - ORDRE IMPORTANT !
cache = SimpleCache()
groq_extractor = GroqLlama3Extractor()

# Créer les instances des nouvelles classes
conversation_manager = AdvancedConversationManager()
intent_analyzer = IntelligentIntentAnalyzer()

# --- FONCTION DE RECHERCHE OPTIMISÉE (CONSERVÉE) ---
def search_properties(query: str) -> Dict[str, Any]:
    """Recherche optimisée avec extraction Llama3 et tracking des tokens - VERSION DEBUG"""
    try:
        logger.info("🚀🚀🚀 DÉBUT FONCTION SEARCH_PROPERTIES 🚀🚀🚀")
        logger.info(f"📝 Requête reçue: '{query}'")
        
        # 1. Vérifier le cache
        logger.info("💾 Vérification du cache...")
        cached_result = cache.get(query)
        if cached_result:
            logger.info(f"🎯 CACHE HIT - Requête servie depuis le cache")
            cached_result["extraction_method"] = "GROQ_LLAMA3"
            return cached_result
        
        logger.info("💾 CACHE MISS - Procédure complète")
        
        # 2. Extraction via Llama3 avec tracking des tokens
        logger.info("🤖 === DÉBUT EXTRACTION GROQ LLAMA3 ===")
        criteria = groq_extractor.extract_criteria(query)
        extraction_method = "GROQ_LLAMA3"
        logger.info("🤖 === FIN EXTRACTION GROQ LLAMA3 ===")
        
        logger.info(f"📋 Critères extraits ({extraction_method}): {criteria}")
        
        # 3. Construction de la requête Elasticsearch
        logger.info("🔧 Construction de la requête Elasticsearch...")
        es_query = ElasticsearchQueryBuilder.build_query(criteria, query)
        logger.info(f"🔧 Requête ES construite: {json.dumps(es_query, indent=2)}")
        
        # 4. Exécution de la recherche
        logger.info("🔍 === DÉBUT RECHERCHE ELASTICSEARCH ===")
        logger.info(f"🔍 Index: {ES_INDEX}")
        logger.info(f"🔍 Query: {es_query}")
        
        res = es.search(index=ES_INDEX, body=es_query)
        logger.info("🔍 === RECHERCHE ELASTICSEARCH TERMINÉE ===")
        
        hits = res["hits"]["hits"]
        total = res["hits"]["total"]["value"] if isinstance(res["hits"]["total"], dict) else res["hits"]["total"]
        
        logger.info(f"📊 Résultats bruts ES: {len(hits)} hits, total: {total}")
        
        # 5. Formatage des résultats
        logger.info("🎨 Formatage des résultats...")
        results = []
        for i, hit in enumerate(hits):
            result = hit["_source"].copy()
            result["score"] = hit["_score"]
            
            if result.get("price") and result.get("area") and result["area"] > 0:
                result["price_per_sqm"] = round(result["price"] / result["area"], 2)
            
            results.append(result)
            logger.info(f"📋 Résultat {i+1}: {result.get('city', 'N/A')} - {result.get('property_type', 'N/A')} - {result.get('price', 'N/A')}DH")
        
        # 6. Log des résultats détaillés
        logger.info(f"✅ === RÉSULTATS FINAUX ===")
        logger.info(f"✅ {len(results)} résultats formatés sur {total} total")
        for i, r in enumerate(results[:3], 1):  # Log des 3 premiers
            logger.info(f"🏠 TOP {i}: {r.get('city', 'N/A')} - {r.get('neighborhood', 'N/A')} - "
                       f"{r.get('price', 'N/A')}DH - {r.get('area', 'N/A')}m² - Score: {r['score']:.2f}")
        
        # 7. Préparer le résultat final
        final_result = {
            "query": query,
            "criteria": criteria,
            "extraction_method": extraction_method,
            "total_results": total,
            "results": results
        }
        
        logger.info(f"✅ Résultat final préparé: {len(results)} propriétés")
        
        # 8. Sauvegarder en cache
        cache.set(query, final_result)
        logger.info("💾 Résultat sauvegardé en cache")
        
        logger.info("🚀🚀🚀 FIN FONCTION SEARCH_PROPERTIES - SUCCÈS 🚀🚀🚀")
        return final_result
        
    except Exception as e:
        logger.error(f"❌❌❌ ERREUR DANS SEARCH_PROPERTIES ❌❌❌")
        logger.error(f"❌ Type erreur: {type(e).__name__}")
        logger.error(f"❌ Message: {str(e)}")
        logger.error(f"❌ Traceback complet:", exc_info=True)
        
        return {
            "query": query,
            "criteria": {},
            "extraction_method": "GROQ_LLAMA3",  # Même en cas d'erreur
            "total_results": 0,
            "results": [],
            "error": str(e)
        }

# --- ORCHESTRATEUR PRINCIPAL ULTRA-INTELLIGENT ---
class UltraIntelligentOrchestrator:
    """Orchestrateur principal avec IA hybride Claude + Groq pour une expérience optimale"""
    
    def __init__(self):
        # Utiliser les instances globales
        self.conversation_manager = conversation_manager
        self.intent_analyzer = intent_analyzer
    
    async def process_message_intelligently(self, session_id: str, user_message: str) -> Dict[str, Any]:
        """Traite un message avec une intelligence artificielle hybride ultra-avancée"""
        
        logger.info("🧠🧠🧠 DÉBUT PROCESSUS INTELLIGENCE HYBRIDE 🧠🧠🧠")
        logger.info(f"🧠 Session: {session_id}")
        logger.info(f"🧠 Message: '{user_message}'")
        
        # ÉTAPE 1: ANALYSE D'INTENT INTELLIGENTE AVEC CLAUDE
        logger.info("🎯 === ÉTAPE 1: ANALYSE D'INTENT CLAUDE ===")
        conversation_history = self.conversation_manager.get_or_create_conversation(session_id)
        
        try:
            intent_analysis = await self.intent_analyzer.analyze_intent_and_decide(
                user_message=user_message,
                conversation_history=conversation_history
            )
            logger.info(f"🎯 INTENT ANALYSÉ: {intent_analysis}")
        except Exception as e:
            logger.error(f"❌ Erreur analyse intent: {str(e)}")
            # Fallback
            intent_analysis = {
                "should_search": True,
                "search_urgency": "immediate",
                "client_qualification_level": "warm",
                "conversation_strategy": "search_and_present"
            }
        
        # ÉTAPE 2: DÉCISION DE RECHERCHE BASÉE SUR L'INTELLIGENCE CLAUDE
        search_results = None
        should_search = intent_analysis.get("should_search", False)
        search_urgency = intent_analysis.get("search_urgency", "none")
        
        if should_search and search_urgency in ["immediate", "progressive"]:
            logger.info("🔍 === ÉTAPE 2: RECHERCHE GROQ + ELASTICSEARCH ===")
            logger.info(f"🔍 Urgence: {search_urgency}")
            logger.info(f"🔍 Niveau client: {intent_analysis.get('client_qualification_level')}")
            
            try:
                search_results = search_properties(user_message)
                logger.info(f"✅ RECHERCHE TERMINÉE: {search_results.get('total_results', 0)} résultats")
                logger.info(f"📋 Critères Groq: {search_results.get('criteria', {})}")
            except Exception as e:
                logger.error(f"❌ Erreur recherche: {str(e)}")
        else:
            logger.info("💬 === CONVERSATION PURE - PAS DE RECHERCHE ===")
            logger.info(f"💬 Raison: should_search={should_search}, urgency={search_urgency}")
        
        # ÉTAPE 3: GÉNÉRATION DE RÉPONSE INTELLIGENTE AVEC CLAUDE
        logger.info("🤖 === ÉTAPE 3: GÉNÉRATION RÉPONSE INTELLIGENTE ===")
        
        try:
            conversational_response = await self.conversation_manager.generate_intelligent_response(
                session_id=session_id,
                user_message=user_message,
                intent_analysis=intent_analysis,
                search_results=search_results
            )
            logger.info("🤖 RÉPONSE GÉNÉRÉE AVEC SUCCÈS")
        except Exception as e:
            logger.error(f"❌ Erreur génération réponse: {str(e)}")
            conversational_response = "Désolé, j'ai rencontré un problème technique."
        
        # ÉTAPE 4: PRÉPARATION RÉPONSE FINALE AVEC MÉTADONNÉES
        logger.info("📦 === ÉTAPE 4: PRÉPARATION RÉPONSE FINALE ===")
        
        # Récupérer le profil client mis à jour
        client_profile = self.conversation_manager.get_or_create_client_profile(session_id)
        
        response = {
            "message": conversational_response,
            "sender": "bot",
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            
            # Métadonnées d'intelligence
            "intent_analysis": intent_analysis,
            "client_profile": client_profile,
            "had_search": should_search,
            "search_urgency": search_urgency,
            
            # Métriques commerciales
            "qualification_level": intent_analysis.get("client_qualification_level"),
            "commercial_intent": intent_analysis.get("commercial_intent"),
            "next_action_suggested": intent_analysis.get("conversation_strategy"),
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
            
            logger.info(f"📊 PROPRIÉTÉS AJOUTÉES: {len(top_results)} top + {len(other_results)} autres")
        
        # Log final du niveau de qualification
        if client_profile["qualification_level"] in ["hot", "ready_to_visit"]:
            logger.info(f"🔥 CLIENT QUALIFIÉ: {client_profile['qualification_level']} - PRÊT POUR VISITE!")
        
        logger.info("🧠🧠🧠 FIN PROCESSUS INTELLIGENCE HYBRIDE - SUCCÈS 🧠🧠🧠")
        return response

# --- API FASTAPI ---
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
        self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict):
        for connection in self.active_connections:
            await connection.send_json(message)

app = FastAPI(title="Chatbot Immobilier Ultra-Intelligent", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instances globales des managers
manager = WebSocketManager()
orchestrator = UltraIntelligentOrchestrator()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    session_id = str(uuid.uuid4())  # Générer un session_id unique pour cette connexion
    logger.info(f"🔌 Nouvelle connexion WebSocket avec session: {session_id}")
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("sender") == "user":
                user_message = message_data.get("message", "")
                
                # Envoi d'un message "en train de traiter..."
                await websocket.send_json({
                    "message": "Youssef analyse votre demande...",
                    "sender": "bot",
                    "timestamp": datetime.now().isoformat(),
                    "session_id": session_id,
                    "is_typing": True
                })
                
                # Traitement avec l'orchestrateur intelligent
                response = await orchestrator.process_message_intelligently(session_id, user_message)
                
                # Envoi de la réponse finale
                await websocket.send_json(response)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info(f"🔌 Déconnexion WebSocket pour session: {session_id}")

@app.post("/chat")
async def chat_endpoint(request: MessageRequest):
    """Endpoint REST pour le chat conversationnel intelligent"""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        response = await orchestrator.process_message_intelligently(session_id, request.message)
        return response
    except Exception as e:
        logger.error(f"❌ Erreur lors du chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur de chat: {str(e)}")

@app.post("/search")
async def search_endpoint(request: QueryRequest):
    """Endpoint REST pour la recherche pure (legacy)"""
    try:
        results = search_properties(request.user_query)
        return results
    except Exception as e:
        logger.error(f"❌ Erreur lors de la recherche: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur de recherche: {str(e)}")

@app.get("/stats")
def get_advanced_stats():
    """Endpoint pour récupérer les statistiques avancées avec intelligence et qualification"""
    
    # Stats quotidiennes avec breakdown par type
    daily_stats = {
        "queries": len(token_tracker.daily_usage),
        "tokens": sum(u.total_tokens for u in token_tracker.daily_usage),
        "cost": sum(u.cost_estimate for u in token_tracker.daily_usage),
        "search_tokens": sum(u.total_tokens for u in token_tracker.daily_usage if u.usage_type == "search"),
        "conversation_tokens": sum(u.total_tokens for u in token_tracker.daily_usage if u.usage_type == "conversation"),
        "intent_analysis_tokens": sum(u.total_tokens for u in token_tracker.daily_usage if u.usage_type == "intent_analysis"),
    }
    
    # Stats des clients par niveau de qualification
    all_profiles = conversation_manager.client_profiles
    qualification_stats = {
        "cold": len([p for p in all_profiles.values() if p["qualification_level"] == "cold"]),
        "warm": len([p for p in all_profiles.values() if p["qualification_level"] == "warm"]),
        "hot": len([p for p in all_profiles.values() if p["qualification_level"] == "hot"]),
        "ready_to_visit": len([p for p in all_profiles.values() if p["qualification_level"] == "ready_to_visit"]),
        "total_qualified": len([p for p in all_profiles.values() if p["qualification_level"] in ["hot", "ready_to_visit"]]),
        "visit_interest": len([p for p in all_profiles.values() if p["visit_interest"] == True])
    }
    
    # Historique des requêtes avec métadonnées d'intelligence
    recent_requests = []
    for i, usage in enumerate(token_tracker.daily_usage[-15:]):
        recent_requests.append({
            "id": i+1,
            "timestamp": usage.timestamp.strftime('%H:%M:%S'),
            "model": usage.model,
            "usage_type": usage.usage_type,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "cost": usage.cost_estimate
        })
    
    # Performance de l'IA
    ai_performance = {
        "total_sessions": len(conversation_manager.conversations),
        "avg_messages_per_session": sum(len(conv) for conv in conversation_manager.conversations.values()) / max(len(conversation_manager.conversations), 1),
        "qualified_conversion_rate": (qualification_stats["total_qualified"] / max(len(all_profiles), 1)) * 100,
        "visit_conversion_rate": (qualification_stats["visit_interest"] / max(len(all_profiles), 1)) * 100
    }
    
    return {
        "daily_stats": daily_stats,
        "total_stats": token_tracker.total_usage,
        "qualification_stats": qualification_stats,
        "ai_performance": ai_performance,
        "cache_stats": {
            "cache_size": len(cache.cache),
            "max_cache_size": cache.max_size
        },
        "conversation_stats": {
            "active_sessions": len(conversation_manager.conversations),
            "total_messages": sum(len(conv) for conv in conversation_manager.conversations.values()),
            "active_websockets": len(manager.active_connections),
            "client_profiles": len(conversation_manager.client_profiles)
        },
        "recent_requests": recent_requests
    }

@app.get("/qualified-leads")
def get_qualified_leads():
    """Endpoint pour les agents commerciaux - Leads qualifiés prêts pour visite"""
    
    qualified_leads = []
    
    for session_id, profile in conversation_manager.client_profiles.items():
        if profile["qualification_level"] in ["hot", "ready_to_visit"] or profile["visit_interest"]:
            
            # Récupérer la dernière conversation
            conversation = conversation_manager.conversations.get(session_id, [])
            last_messages = conversation[-3:] if conversation else []
            
            lead = {
                "session_id": session_id,
                "qualification_level": profile["qualification_level"],
                "visit_interest": profile["visit_interest"],
                "budget_range": profile["budget_range"],
                "preferred_locations": profile["preferred_locations"],
                "property_type": profile["property_type"],
                "interaction_count": profile["interaction_count"],
                "contact_info": profile["contact_info"],
                "last_messages": [
                    {
                        "role": msg["role"],
                        "content": msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                    } for msg in last_messages
                ],
                "priority": "HIGH" if profile["qualification_level"] == "ready_to_visit" else "MEDIUM"
            }
            
            qualified_leads.append(lead)
    
    # Trier par priorité et niveau de qualification
    qualified_leads.sort(key=lambda x: (
        x["qualification_level"] == "ready_to_visit",
        x["visit_interest"],
        x["interaction_count"]
    ), reverse=True)
    
    return {
        "total_qualified_leads": len(qualified_leads),
        "high_priority": len([l for l in qualified_leads if l["priority"] == "HIGH"]),
        "leads": qualified_leads
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "elasticsearch": es.ping(),
        "total_queries": token_tracker.total_usage['total_queries'],
        "total_tokens": token_tracker.total_usage['total_tokens'],
        "active_conversations": len(conversation_manager.conversations),
        "qualified_leads": len([p for p in conversation_manager.client_profiles.values() 
                               if p["qualification_level"] in ["hot", "ready_to_visit"]])
    }

# Route pour la page d'accueil
@app.get("/")
async def read_index():
    """Sert la page d'accueil"""
    from fastapi.responses import FileResponse
    return FileResponse('static/index.html')

# Monter les fichiers statiques sur un chemin spécifique pour éviter les conflits
app.mount("/static", StaticFiles(directory="static"), name="static")

def log_es_index_structure():
    """Fonction pour logger la structure de l'index Elasticsearch"""
    try:
        logger.info("==== [ES] Mapping de l'index '%s' ====" % ES_INDEX)
        mapping = es.indices.get_mapping(index=ES_INDEX)
        # Affiche les champs du mapping
        props = mapping[ES_INDEX]['mappings'].get('properties', {})
        logger.info("Champs de l'index : %s", list(props.keys()))
        for field, desc in props.items():
            logger.info("  - %s: %s", field, desc.get('type', str(desc)))
    except Exception as e:
        logger.error("Erreur lors de la récupération du mapping: %s", str(e))
    try:
        logger.info("==== [ES] Exemples de documents dans '%s' ====" % ES_INDEX)
        res = es.search(index=ES_INDEX, body={"query": {"match_all": {}}, "size": 3})
        for i, hit in enumerate(res['hits']['hits'], 1):
            logger.info("Exemple %d: %s", i, json.dumps(hit['_source'], ensure_ascii=False))
    except Exception as e:
        logger.error("Erreur lors de la récupération des exemples: %s", str(e))

# Appel au démarrage
log_es_index_structure()

def initialize_system():
    """Initialise le système avec vérifications"""
    try:
        logger.info("🔧 === INITIALISATION DU SYSTÈME ===")
        
        # Vérifier Elasticsearch
        if es.ping():
            logger.info("✅ Elasticsearch connecté")
        else:
            logger.error("❌ Elasticsearch non disponible")
        
        # Vérifier Groq
        try:
            test_criteria = groq_extractor.extract_criteria("test")
            logger.info("✅ Groq Llama3 opérationnel")
        except Exception as e:
            logger.error(f"❌ Groq Llama3 erreur: {str(e)}")
        
        # Vérifier Claude
        if ANTHROPIC_API_KEY and ANTHROPIC_API_KEY != "votre_cle_anthropic_ici":
            logger.info("✅ Claude 4 Sonnet configuré")
        else:
            logger.error("❌ Clé Anthropic manquante - Configurez ANTHROPIC_API_KEY")
        
        logger.info("🔧 === SYSTÈME INITIALISÉ ===")
        
    except Exception as e:
        logger.error(f"❌ Erreur initialisation: {str(e)}")

# Initialiser le système
initialize_system()

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Démarrage du chatbot conversationnel ultra-intelligent")
    logger.info("🧠 Claude 4 Sonnet + Groq Llama3 + Intelligence Hybride")
    logger.info("🎯 Système de qualification client automatique")
    uvicorn.run(app, host="0.0.0.0", port=8000)

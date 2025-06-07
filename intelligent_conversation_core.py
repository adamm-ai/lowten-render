# ========================================================================================
# SYSTÈME DE CONVERSATION IMMOBILIÈRE INTELLIGENT ET ORIENTÉ OBJECTIFS
# Version optimisée - User-friendly avec progression naturelle vers la conversion
# ========================================================================================

import json
import logging
import re
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)

# ========================================================================================
# MÉMOIRE CONVERSATIONNELLE ORIENTÉE OBJECTIFS
# ========================================================================================

@dataclass
class ObjectiveMemory:
    """Mémoire orientée vers l'atteinte d'objectifs de conversion"""
    
    # Informations client essentielles pour la conversion
    name: Optional[str] = None
    phone: Optional[str] = None 
    email: Optional[str] = None
    
    # Critères de recherche (collectés naturellement)
    budget_max: Optional[int] = None
    budget_min: Optional[int] = None
    locations_preferred: List[str] = field(default_factory=list)
    property_type: Optional[str] = None
    bedrooms_min: Optional[int] = None
    area_min: Optional[int] = None
    
    # État de progression vers l'objectif
    current_objective: str = "discover_needs"  # discover_needs -> search_properties -> present_details -> qualify_lead -> close_conversion
    properties_shown: List[Dict] = field(default_factory=list)
    focused_property: Optional[Dict] = None
    interest_signals: List[str] = field(default_factory=list)
    conversion_readiness: float = 0.0
    
    # Contexte conversationnel
    conversation_flow: str = "exploration"  # exploration -> qualification -> conversion
    client_satisfaction: str = "neutral"  # positive, neutral, frustrated
    last_action: Optional[str] = None
    
    def get_completion_score(self) -> float:
        """Score de complétude des informations collectées"""
        score = 0.0
        
        # Critères de recherche (60% du score)
        if self.budget_max: score += 0.2
        if self.locations_preferred: score += 0.2
        if self.property_type: score += 0.1
        if self.bedrooms_min: score += 0.1
        
        # Informations de contact (40% du score)
        if self.name: score += 0.15
        if self.phone or self.email: score += 0.25
        
        return min(1.0, score)
    
    def is_ready_for_conversion(self) -> bool:
        """Détermine si le client est prêt pour la conversion"""
        return (
            self.get_completion_score() >= 0.6 and
            self.focused_property is not None and
            len(self.interest_signals) >= 2 and
            self.client_satisfaction in ["positive", "neutral"]
        )
    
    def add_interest_signal(self, signal: str):
        """Ajoute un signal d'intérêt"""
        if signal not in self.interest_signals:
            self.interest_signals.append(signal)
            self.conversion_readiness = min(1.0, self.conversion_readiness + 0.2)

# ========================================================================================
# DÉTECTEUR D'INTENTIONS ET SIGNAUX D'INTÉRÊT
# ========================================================================================

class IntentDetector:
    """Détecte les intentions et signaux d'intérêt du client"""
    
    def __init__(self):
        # Signaux d'intérêt positifs
        self.interest_signals = {
            "property_details_request": [
                "détails", "plus d'info", "en savoir plus", "caractéristiques",
                "première", "deuxième", "celui", "cette propriété", "cet appartement"
            ],
            "visit_interest": [
                "visite", "visiter", "voir", "rendez-vous", "disponible",
                "quand peut-on", "organiser", "planifier"
            ],
            "contact_request": [
                "contact", "téléphone", "appeler", "agent", "propriétaire",
                "coordonnées", "joindre"
            ],
            "positive_feedback": [
                "intéressant", "bien", "parfait", "me plaît", "j'aime",
                "super", "génial", "exactement", "correspond"
            ],
            "budget_confirmation": [
                "dans mes moyens", "ça va", "budget ok", "prix correct",
                "abordable", "faisable"
            ]
        }
        
        # Signaux de frustration/désintérêt
        self.negative_signals = [
            "trop cher", "hors budget", "pas intéressé", "non merci",
            "autre chose", "différent", "ne me convient pas"
        ]
        
        # Déclencheurs de recherche
        self.search_triggers = [
            "voir", "montrer", "proposer", "options", "disponible",
            "qu'est-ce que vous avez", "autres propriétés"
        ]
    
    def detect_intent_and_signals(self, message: str, memory: ObjectiveMemory) -> Dict[str, Any]:
        """Détecte l'intention et les signaux d'intérêt"""
        
        message_lower = message.lower()
        detected_signals = []
        primary_intent = "general_inquiry"
        action_required = None
        
        # Détecter les signaux d'intérêt
        for signal_type, keywords in self.interest_signals.items():
            if any(keyword in message_lower for keyword in keywords):
                detected_signals.append(signal_type)
                
                if signal_type == "property_details_request":
                    primary_intent = "request_property_details"
                    action_required = "show_property_details"
                elif signal_type == "visit_interest":
                    primary_intent = "request_visit"
                    action_required = "organize_visit"
                elif signal_type == "contact_request":
                    primary_intent = "request_contact"
                    action_required = "provide_contact_info"
        
        # Détecter demande de recherche
        if any(trigger in message_lower for trigger in self.search_triggers):
            if not memory.properties_shown or "autre" in message_lower:
                primary_intent = "request_search"
                action_required = "perform_search"
        
        # Détecter signaux négatifs
        negative_detected = any(signal in message_lower for signal in self.negative_signals)
        
        return {
            "primary_intent": primary_intent,
            "interest_signals": detected_signals,
            "action_required": action_required,
            "negative_sentiment": negative_detected,
            "should_search": action_required == "perform_search",
            "should_qualify": len(detected_signals) >= 2,
            "conversion_opportunity": primary_intent in ["request_visit", "request_contact"]
        }

# ========================================================================================
# EXTRACTEUR D'INFORMATIONS INTELLIGENT
# ========================================================================================

class SmartInfoExtractor:
    """Extrait les informations de manière intelligente et non intrusive"""
    
    def __init__(self, groq_client, model: str = "llama3-70b-8192"):
        self.groq_client = groq_client
        self.model = model
        
        # Patterns d'extraction
        self.extraction_patterns = {
            "budget": [
                r'(\d+(?:\.\d+)?)\s*(?:millions?|m)\s*(?:mad|dh|dirhams?)?',
                r'budget.*?(\d+(?:\.\d+)?)\s*(?:millions?|m)',
                r'(\d+)\s*(?:à|jusqu\'à|-)\s*(\d+)\s*(?:millions?|m)'
            ],
            "locations": [
                r'\b(?:à|dans|vers|côté|près de|secteur|quartier)\s+([A-Za-zÀ-ÿ\s\-\']{3,30})',
                r'\b([A-Za-zÀ-ÿ\s\-\']{3,25})\s+(?:me plaît|m\'intéresse|c\'est bien)'
            ],
            "bedrooms": [
                r'(\d+)\s*(?:chambres?|ch\b)',
                r'(\d+)\s*pièces?'
            ],
            "contact": [
                r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b',  # Nom complet
                r'\b(\d{10}|\+212\d{9})\b',        # Téléphone
                r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b'  # Email
            ]
        }
    
    async def extract_info_naturally(self, message: str, memory: ObjectiveMemory) -> ObjectiveMemory:
        """Extrait les informations naturellement sans forcer"""
        
        # Extraction par patterns
        pattern_extracted = self._extract_with_patterns(message)
        
        # Extraction contextuelle avec LLM si nécessaire
        llm_extracted = {}
        if self._needs_llm_extraction(message, pattern_extracted):
            llm_extracted = await self._extract_with_llm(message, memory)
        
        # Fusion et mise à jour de la mémoire
        return self._update_memory_smartly(memory, pattern_extracted, llm_extracted, message)
    
    def _extract_with_patterns(self, message: str) -> Dict[str, Any]:
        """Extraction rapide par patterns regex"""
        
        extracted = {}
        message_lower = message.lower()
        
        # Budget
        for pattern in self.extraction_patterns["budget"]:
            match = re.search(pattern, message_lower)
            if match:
                if len(match.groups()) == 2:  # Range
                    extracted["budget_min"] = int(float(match.group(1)) * 1000000)
                    extracted["budget_max"] = int(float(match.group(2)) * 1000000)
                else:  # Single value
                    extracted["budget_max"] = int(float(match.group(1)) * 1000000)
                break
        
        # Locations
        locations = []
        for pattern in self.extraction_patterns["locations"]:
            matches = re.findall(pattern, message_lower, re.IGNORECASE)
            for match in matches:
                location = match if isinstance(match, str) else match[1] if len(match) > 1 else match[0]
                location = location.strip()
                if len(location) > 2 and location not in ['les', 'des', 'aux']:
                    locations.append(location.title())
        
        if locations:
            extracted["locations"] = locations
        
        # Chambres
        for pattern in self.extraction_patterns["bedrooms"]:
            match = re.search(pattern, message_lower)
            if match:
                extracted["bedrooms_min"] = int(match.group(1))
                break
        
        # Contact (nom, téléphone, email)
        contact_match = re.search(self.extraction_patterns["contact"][0], message)
        if contact_match:
            extracted["name"] = contact_match.group(1)
        
        phone_match = re.search(self.extraction_patterns["contact"][1], message)
        if phone_match:
            extracted["phone"] = phone_match.group(1)
        
        email_match = re.search(self.extraction_patterns["contact"][2], message)
        if email_match:
            extracted["email"] = email_match.group(1)
        
        return extracted
    
    def _needs_llm_extraction(self, message: str, pattern_extracted: Dict) -> bool:
        """Détermine si l'extraction LLM est nécessaire"""
        return (
            len(message.split()) > 10 and  # Message suffisamment long
            not pattern_extracted and      # Patterns n'ont rien trouvé
            any(word in message.lower() for word in ['cherche', 'veux', 'besoin', 'préfère'])
        )
    
    async def _extract_with_llm(self, message: str, memory: ObjectiveMemory) -> Dict[str, Any]:
        """Extraction contextuelle avec LLM"""
        
        current_info = f"""
        Budget actuel: {memory.budget_max or 'Non défini'}
        Locations: {', '.join(memory.locations_preferred) or 'Non définies'}
        Type: {memory.property_type or 'Non défini'}
        Chambres: {memory.bedrooms_min or 'Non défini'}
        """
        
        prompt = f"""Extrait UNIQUEMENT les nouvelles informations explicites de ce message client.

MESSAGE: "{message}"

CONTEXTE ACTUEL: {current_info}

Réponds en JSON valide avec SEULEMENT les nouveaux éléments mentionnés:
{{
    "budget_max": null,
    "budget_min": null,
    "locations": [],
    "property_type": null,
    "bedrooms_min": null,
    "area_min": null,
    "name": null,
    "phone": null,
    "email": null
}}

N'invente RIEN. Si pas mentionné explicitement = null/[]"""

        try:
            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            result = response.choices[0].message.content.strip()
            if result.startswith('```json'):
                result = result[7:-3]
            elif result.startswith('```'):
                result = result[3:-3]
            
            return json.loads(result)
            
        except Exception as e:
            logger.error(f"Erreur extraction LLM: {e}")
            return {}
    
    def _update_memory_smartly(self, memory: ObjectiveMemory, pattern_data: Dict, llm_data: Dict, original_message: str) -> ObjectiveMemory:
        """Met à jour la mémoire intelligemment"""
        
        # Fusionner les données extraites
        all_extracted = {**pattern_data, **llm_data}
        
        # Mise à jour sélective (ne pas écraser si déjà défini et cohérent)
        if all_extracted.get("budget_max") and not memory.budget_max:
            memory.budget_max = all_extracted["budget_max"]
        
        if all_extracted.get("budget_min") and not memory.budget_min:
            memory.budget_min = all_extracted["budget_min"]
        
        # Locations - ajouter sans doublons
        if all_extracted.get("locations"):
            for location in all_extracted["locations"]:
                if location not in memory.locations_preferred:
                    memory.locations_preferred.append(location)
        
        # Autres critères
        if all_extracted.get("property_type") and not memory.property_type:
            memory.property_type = all_extracted["property_type"]
        
        if all_extracted.get("bedrooms_min") and not memory.bedrooms_min:
            memory.bedrooms_min = all_extracted["bedrooms_min"]
        
        if all_extracted.get("area_min") and not memory.area_min:
            memory.area_min = all_extracted["area_min"]
        
        # Informations de contact
        if all_extracted.get("name") and not memory.name:
            memory.name = all_extracted["name"]
        
        if all_extracted.get("phone") and not memory.phone:
            memory.phone = all_extracted["phone"]
        
        if all_extracted.get("email") and not memory.email:
            memory.email = all_extracted["email"]
        
        return memory

# ========================================================================================
# GÉNÉRATEUR DE REQUÊTES DE RECHERCHE INTELLIGENT
# ========================================================================================

class IntelligentSearchQueryBuilder:
    """Construit des requêtes de recherche optimisées pour trouver ce que le client veut vraiment"""
    
    def __init__(self):
        self.priority_weights = {
            "user_message": 1.0,      # Le message actuel est prioritaire
            "budget": 0.8,            # Budget très important
            "location": 0.7,          # Location importante
            "property_type": 0.6,     # Type de propriété
            "bedrooms": 0.5,          # Chambres
            "area": 0.4               # Surface
        }
    
    def build_search_query(self, user_message: str, memory: ObjectiveMemory, intent_analysis: Dict) -> str:
        """Construit une requête de recherche intelligente"""
        
        query_parts = []
        
        # 1. Message utilisateur nettoyé (priorité absolue)
        cleaned_message = self._clean_user_message(user_message)
        if cleaned_message:
            query_parts.append(f"{cleaned_message}")
        
        # 2. Budget si défini et pertinent
        if memory.budget_max:
            if memory.budget_min:
                query_parts.append(f"entre {memory.budget_min:,} et {memory.budget_max:,} MAD")
            else:
                query_parts.append(f"maximum {memory.budget_max:,} MAD")
        
        # 3. Locations préférées (max 2)
        if memory.locations_preferred:
            top_locations = memory.locations_preferred[:2]
            query_parts.append(f"à {' ou '.join(top_locations)}")
        
        # 4. Type de propriété
        if memory.property_type:
            query_parts.append(memory.property_type)
        
        # 5. Chambres
        if memory.bedrooms_min:
            query_parts.append(f"{memory.bedrooms_min} chambres minimum")
        
        # 6. Surface si spécifiée
        if memory.area_min:
            query_parts.append(f"{memory.area_min}m² minimum")
        
        # Assemblage intelligent
        final_query = " ".join(query_parts)
        
        # Optimisation selon l'intention
        if intent_analysis.get("action_required") == "perform_search":
            # Recherche plus large si demande explicite
            final_query = self._broaden_search_if_needed(final_query, memory)
        
        return final_query[:200]  # Limiter la longueur
    
    def _clean_user_message(self, message: str) -> str:
        """Nettoie le message pour la recherche"""
        
        # Mots à supprimer
        noise_words = [
            'bonjour', 'bonsoir', 'salut', 'merci', 'svp', 's\'il vous plaît',
            'je veux', 'je cherche', 'j\'aimerais', 'pouvez-vous', 'est-ce que',
            'qu\'est-ce que', 'montrez-moi', 'voir', 'regarder'
        ]
        
        cleaned = message.lower()
        for noise in noise_words:
            cleaned = re.sub(r'\b' + re.escape(noise) + r'\b', '', cleaned)
        
        # Garder les mots significatifs
        words = [word.strip() for word in cleaned.split() if len(word.strip()) > 2]
        return " ".join(words[:8])  # Max 8 mots
    
    def _broaden_search_if_needed(self, query: str, memory: ObjectiveMemory) -> str:
        """Élargit la recherche si nécessaire"""
        
        # Si recherche précédente n'a pas donné de résultats satisfaisants
        if memory.last_action == "search_performed" and memory.client_satisfaction == "frustrated":
            # Enlever certains critères pour élargir
            query = re.sub(r'\d+ chambres minimum', '', query)
            query = re.sub(r'\d+m² minimum', '', query)
        
        return query.strip()

# ========================================================================================
# GÉNÉRATEUR DE RÉPONSES ORIENTÉ OBJECTIFS
# ========================================================================================

class ObjectiveResponseGenerator:
    """Génère des réponses orientées vers l'atteinte d'objectifs de conversion"""
    
    def __init__(self, groq_client, model: str = "llama3-70b-8192"):
        self.groq_client = groq_client
        self.model = model
        
        # Templates orientés objectifs
        self.objective_templates = {
            "discover_needs": {
                "no_criteria": "Parfait ! Qu'est-ce qui vous amène à chercher une nouvelle propriété ?",
                "partial_criteria": "Super, on avance ! {context}",
                "with_search": "Excellent ! Voici ce que j'ai trouvé pour vous :"
            },
            "search_properties": {
                "presenting_results": "J'ai trouvé {count} propriétés qui correspondent à vos critères :",
                "no_results": "Je n'ai pas trouvé de correspondance exacte, mais voici des options similaires :",
                "follow_up": "Laquelle vous intéresse le plus ?"
            },
            "present_details": {
                "property_details": "Voici tous les détails sur cette propriété :",
                "highlight_benefits": "Ce qui rend cette propriété spéciale :",
                "gauge_interest": "Qu'est-ce qui vous plaît le plus ?"
            },
            "qualify_lead": {
                "visit_opportunity": "Souhaiteriez-vous organiser une visite ?",
                "contact_collection": "Pour organiser cela, j'aurais besoin de vos coordonnées.",
                "timeline_check": "Quand seriez-vous disponible ?"
            },
            "close_conversion": {
                "visit_confirmation": "Parfait ! Je vais organiser votre visite.",
                "contact_confirmation": "Merci ! L'agent vous contactera sous 24h.",
                "next_steps": "Voici les prochaines étapes :"
            }
        }
    
    async def generate_objective_response(self, 
                                        user_message: str,
                                        memory: ObjectiveMemory,
                                        intent_analysis: Dict,
                                        search_results: Optional[List] = None) -> str:
        """Génère une réponse orientée objectif"""
        
        # Déterminer l'objectif actuel
        current_objective = self._determine_current_objective(memory, intent_analysis)
        memory.current_objective = current_objective
        
        # Contexte pour le LLM
        context = self._build_objective_context(user_message, memory, intent_analysis, search_results, current_objective)
        
        # Générer avec LLM
        try:
            response = await self._generate_with_objective_llm(context)
            return self._polish_objective_response(response, memory, current_objective)
        except Exception as e:
            logger.error(f"Erreur génération objective: {e}")
            return self._get_objective_fallback(memory, current_objective)
    
    def _determine_current_objective(self, memory: ObjectiveMemory, intent_analysis: Dict) -> str:
        """Détermine l'objectif actuel basé sur l'état de la conversation - VERSION AGRESSIVE"""
        
        # NOUVEAU: Recherche immédiate si critères minimaux
        has_search_criteria = (
            memory.locations_preferred or 
            memory.property_type or 
            memory.bedrooms_min or
            memory.budget_max
        )
        
        # Conversion immédiate si ready
        if memory.is_ready_for_conversion() and intent_analysis.get("conversion_opportunity"):
            return "close_conversion"
        
        # Qualification si signaux d'intérêt élevés
        if len(memory.interest_signals) >= 2 or intent_analysis.get("should_qualify"):
            return "qualify_lead"
        
        # Présentation de détails si propriété focalisée
        if memory.focused_property or intent_analysis.get("action_required") == "show_property_details":
            return "present_details"
        
        # MODIFIÉ: Recherche plus agressive
        if (intent_analysis.get("should_search") or 
            has_search_criteria or  # NOUVEAU: critères minimaux suffisent
            any(word in intent_analysis.get("primary_intent", "") for word in ["request_search", "general_inquiry"])):
            return "search_properties"
        
        # Découverte seulement si vraiment aucun critère
        return "discover_needs"
    
    def _build_objective_context(self, message: str, memory: ObjectiveMemory, intent: Dict, results: Optional[List], objective: str) -> str:
        """Construit le contexte pour la génération orientée objectif"""
        
        context_parts = [
            f"MESSAGE CLIENT: \"{message}\"",
            f"OBJECTIF ACTUEL: {objective}",
            f"INTENTION DÉTECTÉE: {intent.get('primary_intent', 'general_inquiry')}",
            f"SIGNAUX D'INTÉRÊT: {', '.join(intent.get('interest_signals', []))}"
        ]
        
        # Informations collectées
        info_summary = []
        if memory.budget_max:
            info_summary.append(f"Budget: {memory.budget_max:,} MAD")
        if memory.locations_preferred:
            info_summary.append(f"Secteurs: {', '.join(memory.locations_preferred[:2])}")
        if memory.property_type:
            info_summary.append(f"Type: {memory.property_type}")
        if memory.bedrooms_min:
            info_summary.append(f"Chambres: {memory.bedrooms_min}+")
        
        if info_summary:
            context_parts.append(f"CRITÈRES COLLECTÉS: {' | '.join(info_summary)}")
        
        # Informations de contact
        contact_info = []
        if memory.name:
            contact_info.append(f"Nom: {memory.name}")
        if memory.phone:
            contact_info.append(f"Tél: {memory.phone}")
        if memory.email:
            contact_info.append(f"Email: {memory.email}")
        
        if contact_info:
            context_parts.append(f"CONTACT: {' | '.join(contact_info)}")
        
        # Résultats de recherche si disponibles
        if results:
            results_summary = []
            for i, prop in enumerate(results[:3], 1):
                location = f"{prop.get('neighborhood', 'N/A')}, {prop.get('city', 'N/A')}"
                price = f"{prop.get('price', 'N/A'):,}" if isinstance(prop.get('price'), (int, float)) else str(prop.get('price', 'N/A'))
                area = prop.get('area', 'N/A')
                bedrooms = prop.get('bedrooms', 'N/A')
                results_summary.append(f"{i}. {location} - {price} MAD - {area}m² - {bedrooms}ch")
            
            context_parts.append(f"PROPRIÉTÉS TROUVÉES:\n" + "\n".join(results_summary))
        
        # État de conversion
        context_parts.append(f"COMPLÉTUDE INFO: {memory.get_completion_score():.1f}")
        context_parts.append(f"PRÊT CONVERSION: {'OUI' if memory.is_ready_for_conversion() else 'NON'}")
        
        return "\n\n".join(context_parts)
    
    async def _generate_with_objective_llm(self, context: str) -> str:
        """Génération LLM orientée objectifs"""
        
        system_prompt = """Tu es un conseiller immobilier expert orienté RÉSULTATS et ACTION IMMÉDIATE.

RÈGLES ABSOLUES:
1. Si tu as des propriétés à montrer, MONTRE-LES IMMÉDIATEMENT
2. Si le client demande des options, PRÉSENTE-LES sans poser de questions
3. Sois enthousiaste et direct : "Voici ce que j'ai trouvé pour vous !"
4. Maximum UNE question par réponse, et SEULEMENT si vraiment nécessaire
5. Privilégie TOUJOURS l'action à la conversation
6. Crée de l'excitation autour des propriétés

PRIORITÉS:
1. MONTRER des propriétés = priorité #1
2. Collecter infos de contact si intérêt détecté
3. Organiser visites
4. CONVERTIR

ÉVITE ABSOLUMENT:
- Les questions inutiles quand tu as des résultats
- Les "j'ai besoin de savoir..." 
- Les conversations sans fin
- Poser des questions quand le client veut voir des options

STYLE:
- Direct et efficace
- Enthousiaste sur les propriétés 
- Orienté action immédiate"""

        user_prompt = f"""{context}

GÉNÈRE UNE RÉPONSE qui:
1. Répond parfaitement au client
2. Progresse vers l'objectif actuel
3. Reste naturelle et engageante
4. Inclut une proposition d'action si approprié
5. Crée de l'enthousiasme

Réponse:"""

        response = self.groq_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=350
        )
        
        return response.choices[0].message.content.strip()
    
    def _polish_objective_response(self, response: str, memory: ObjectiveMemory, objective: str) -> str:
        """Peaufine la réponse selon l'objectif"""
        
        # Ajouter des éléments selon l'objectif
        if objective == "search_properties" and memory.budget_max:
            if "budget" not in response.lower():
                response += f" (Dans votre budget de {memory.budget_max:,} MAD)"
        
        if objective == "qualify_lead" and not any(word in response.lower() for word in ['visite', 'contact', 'rendez-vous']):
            response += " Souhaitez-vous organiser une visite ?"
        
        if objective == "close_conversion" and memory.name and not memory.phone:
            response += " J'aurais juste besoin de votre numéro pour finaliser."
        
        return response.strip()
    
    def _get_objective_fallback(self, memory: ObjectiveMemory, objective: str) -> str:
        """Fallback orienté objectif"""
        
        templates = self.objective_templates.get(objective, {})
        
        if objective == "discover_needs":
            if memory.get_completion_score() < 0.3:
                return "Parfait ! Dites-moi ce que vous recherchez comme propriété."
            else:
                return "Super ! Continuons à affiner votre recherche."
        
        elif objective == "search_properties":
            return "Laissez-moi vous proposer quelques options intéressantes."
        
        elif objective == "present_details":
            return "Cette propriété a de très bons atouts ! Qu'est-ce qui vous intéresse le plus ?"
        
        elif objective == "qualify_lead":
            return "Cette propriété vous plaît ? On peut organiser une visite si vous voulez."
        
        elif objective == "close_conversion":
            return "Parfait ! Je vais tout organiser pour vous."
        
        return "Comment puis-je vous aider avec votre recherche ?"

# ========================================================================================
# GESTIONNAIRE PRINCIPAL ORIENTÉ OBJECTIFS
# ========================================================================================

class ModularConversationManagerEnhanced:
    """Gestionnaire principal orienté objectifs et conversion"""
    
    def __init__(self, groq_client, model: str = "llama3-70b-8192"):
        self.groq_client = groq_client
        self.model = model
        
        # Composants intelligents
        self.intent_detector = IntentDetector()
        self.info_extractor = SmartInfoExtractor(groq_client, model)
        self.query_builder = IntelligentSearchQueryBuilder()
        self.response_generator = ObjectiveResponseGenerator(groq_client, model)
        
        # Mémoire conversationnelle
        self.conversation_memory = ObjectiveMemory()
        self.conversation_history = []
        self.message_count = 0
        self.conversation_start = datetime.now()
        
        logger.info("🎯 Gestionnaire orienté objectifs initialisé")
    
    async def process_message(self, user_message: str, search_function: callable = None) -> Dict[str, Any]:
        """Traite un message avec orientation objectifs"""
        
        try:
            self.message_count += 1
            processing_start = datetime.now()
            
            logger.info(f"🎯 Message #{self.message_count}: {user_message[:100]}...")
            
            # 1. DÉTECTION D'INTENTION ET SIGNAUX D'INTÉRÊT
            intent_analysis = self.intent_detector.detect_intent_and_signals(user_message, self.conversation_memory)
            
            # Ajouter les signaux d'intérêt détectés
            for signal in intent_analysis.get("interest_signals", []):
                self.conversation_memory.add_interest_signal(signal)
            
            logger.info(f"🎯 Intention: {intent_analysis['primary_intent']}, Signaux: {intent_analysis['interest_signals']}")
            
            # 2. EXTRACTION INTELLIGENTE D'INFORMATIONS
            self.conversation_memory = await self.info_extractor.extract_info_naturally(
                user_message, self.conversation_memory
            )
            
            # 3. GESTION DE LA PROPRIÉTÉ FOCALISÉE
            focused_property_updated = self._handle_property_focus(user_message, intent_analysis)
            
            # 4. DÉCISION DE RECHERCHE ULTRA-AGRESSIVE
            search_decision = self._should_search_aggressively(user_message, self.conversation_memory)
            
            search_results = None
            if search_decision["should_search"] and search_function:
                search_query = self.query_builder.build_search_query(
                    user_message, self.conversation_memory, intent_analysis
                )
                
                logger.info(f"🔍 Recherche AGRESSIVE: {search_query}")
                search_response = search_function(search_query)
                search_results = search_response.get("results", []) if search_response else []
                
                # Sauvegarder les propriétés montrées
                if search_results:
                    self.conversation_memory.properties_shown.extend(search_results[:5])
                    self.conversation_memory.last_action = "search_performed"
                
                logger.info(f"✅ {len(search_results)} propriétés trouvées et présentées")
            
            # Si pas de résultats mais recherche demandée, essayer recherche élargie
            elif search_decision["should_search"] and not search_results and search_function:
                # Recherche élargie avec moins de critères
                broader_query = self._build_broader_search_query(user_message, self.conversation_memory)
                logger.info(f"🔍 Recherche ÉLARGIE: {broader_query}")
                
                search_response = search_function(broader_query)
                search_results = search_response.get("results", []) if search_response else []
                
                if search_results:
                    self.conversation_memory.properties_shown.extend(search_results[:5])
                    logger.info(f"✅ Recherche élargie: {len(search_results)} propriétés trouvées")
            
            # 5. GÉNÉRATION DE RÉPONSE ORIENTÉE OBJECTIFS
            response_message = await self.response_generator.generate_objective_response(
                user_message, self.conversation_memory, intent_analysis, search_results
            )
            
            # 6. MISE À JOUR DE L'ÉTAT CONVERSATIONNEL
            self._update_conversation_state(user_message, intent_analysis, search_results)
            
            # 7. MISE À JOUR HISTORIQUE
            self.conversation_history.append(f"Client: {user_message}")
            self.conversation_history.append(f"Agent: {response_message}")
            
            # Limiter l'historique
            if len(self.conversation_history) > 12:
                self.conversation_history = self.conversation_history[-12:]
            
            # 8. CALCUL DES MÉTRIQUES DE PERFORMANCE
            processing_time = (datetime.now() - processing_start).total_seconds()
            
            # 9. CONSTRUCTION DE LA RÉPONSE FINALE
            return self._build_final_response(
                response_message, intent_analysis, search_results, 
                search_response if search_results else None, processing_time
            )
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement message: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            
            return self._get_error_fallback(user_message)
    
    def _handle_property_focus(self, user_message: str, intent_analysis: Dict) -> bool:
        """Gère la focalisation sur une propriété spécifique"""
        
        if intent_analysis.get("action_required") == "show_property_details":
            # Identifier la propriété référencée
            referenced_property = self._identify_referenced_property(user_message)
            
            if referenced_property:
                self.conversation_memory.focused_property = referenced_property
                self.conversation_memory.add_interest_signal("property_details_request")
                logger.info(f"🎯 Focus sur propriété: {referenced_property.get('id', 'N/A')}")
                return True
        
        return False
    
    def _identify_referenced_property(self, message: str) -> Optional[Dict]:
        """Identifie la propriété référencée dans le message"""
        
        if not self.conversation_memory.properties_shown:
            return None
        
        message_lower = message.lower()
        
        # Références ordinales
        ordinal_patterns = {
            1: ["premier", "première", "1er", "1ère", "premier appartement"],
            2: ["deuxième", "deuxieme", "second", "seconde", "2ème", "2eme"],
            3: ["troisième", "troisieme", "3ème", "3eme"],
            4: ["quatrième", "quatrieme", "4ème", "4eme"],
            5: ["cinquième", "cinquieme", "5ème", "5eme"]
        }
        
        for position, patterns in ordinal_patterns.items():
            if any(pattern in message_lower for pattern in patterns):
                if len(self.conversation_memory.properties_shown) >= position:
                    return self.conversation_memory.properties_shown[position - 1]
        
        # Références contextuelles
        context_patterns = [
            "cette propriété", "cet appartement", "cette maison", "ce bien",
            "celui-ci", "celle-ci", "celui-là", "celle-là"
        ]
        
        if any(pattern in message_lower for pattern in context_patterns):
            # Retourner la première propriété par défaut
            return self.conversation_memory.properties_shown[0] if self.conversation_memory.properties_shown else None
        
        return None
    
    def _update_conversation_state(self, user_message: str, intent_analysis: Dict, search_results: Optional[List]):
        """Met à jour l'état conversationnel"""
        
        # Mettre à jour la satisfaction client
        if intent_analysis.get("negative_sentiment"):
            self.conversation_memory.client_satisfaction = "frustrated"
        elif any(signal in intent_analysis.get("interest_signals", []) for signal in ["positive_feedback", "visit_interest"]):
            self.conversation_memory.client_satisfaction = "positive"
        else:
            self.conversation_memory.client_satisfaction = "neutral"
        
        # Mettre à jour le flow conversationnel
        if self.conversation_memory.get_completion_score() >= 0.6:
            self.conversation_memory.conversation_flow = "qualification"
        elif self.conversation_memory.get_completion_score() >= 0.8:
            self.conversation_memory.conversation_flow = "conversion"
        
        # Calculer la readiness de conversion
        self.conversation_memory.conversion_readiness = self._calculate_conversion_readiness()
    
    def _calculate_conversion_readiness(self) -> float:
        """Calcule la disposition à la conversion"""
        
        score = 0.0
        
        # Informations collectées (40%)
        score += self.conversation_memory.get_completion_score() * 0.4
        
        # Signaux d'intérêt (30%)
        interest_score = min(1.0, len(self.conversation_memory.interest_signals) / 4)
        score += interest_score * 0.3
        
        # Propriété focalisée (20%)
        if self.conversation_memory.focused_property:
            score += 0.2
        
        # Satisfaction client (10%)
        satisfaction_scores = {"positive": 0.1, "neutral": 0.05, "frustrated": 0.0}
        score += satisfaction_scores.get(self.conversation_memory.client_satisfaction, 0.0)
        
        return min(1.0, score)
    
    def _build_final_response(self, message: str, intent_analysis: Dict, search_results: Optional[List], 
                            search_response: Optional[Dict], processing_time: float) -> Dict[str, Any]:
        """Construit la réponse finale orientée objectifs"""
        
        # Format client_info compatible
        client_info = {
            "name": self.conversation_memory.name,
            "phone": self.conversation_memory.phone,
            "email": self.conversation_memory.email,
            "budget_min": self.conversation_memory.budget_min,
            "budget_max": self.conversation_memory.budget_max,
            "preferred_locations": self.conversation_memory.locations_preferred,
            "property_type": self.conversation_memory.property_type,
            "bedrooms_min": self.conversation_memory.bedrooms_min,
            "move_in_timeline": None,  # Pas encore implémenté
            "financing_status": None,
            "specific_requirements": [],
            "viewing_preference": None
        }
        
        # Statistiques de conversation
        duration = (datetime.now() - self.conversation_start).total_seconds() / 60
        conversation_stats = {
            "message_count": self.message_count,
            "duration_minutes": round(duration, 2),
            "objective": self.conversation_memory.current_objective,
            "conversion_readiness": self.conversation_memory.conversion_readiness,
            "completion_score": self.conversation_memory.get_completion_score(),
            "interest_signals_count": len(self.conversation_memory.interest_signals),
            "properties_shown": len(self.conversation_memory.properties_shown),
            "focused_property": bool(self.conversation_memory.focused_property)
        }
        
        # Phase et niveau d'intérêt pour compatibilité
        phase = self._map_objective_to_phase(self.conversation_memory.current_objective)
        interest_level = self._map_readiness_to_interest(self.conversation_memory.conversion_readiness)
        
        return {
            "message": message,
            "phase": phase,
            "interest_level": interest_level,
            "client_info": client_info,
            "search_results": search_response,
            "conversation_stats": conversation_stats,
            "intent_analysis": intent_analysis,
            "memory_insights": {
                "current_objective": self.conversation_memory.current_objective,
                "conversion_readiness": self.conversation_memory.conversion_readiness,
                "completion_score": self.conversation_memory.get_completion_score(),
                "interest_signals": self.conversation_memory.interest_signals,
                "client_satisfaction": self.conversation_memory.client_satisfaction,
                "focused_property": self.conversation_memory.focused_property,
                "ready_for_conversion": self.conversation_memory.is_ready_for_conversion(),
                "next_recommended_action": self._get_next_recommended_action()
            },
            "performance_metrics": {
                "processing_time_ms": processing_time * 1000,
                "search_performed": bool(search_results),
                "information_extracted": bool(intent_analysis.get("interest_signals"))
            }
        }
    
    def _map_objective_to_phase(self, objective: str) -> str:
        """Mappe les objectifs vers les phases compatibles"""
        mapping = {
            "discover_needs": "exploration",
            "search_properties": "search",
            "present_details": "evaluation",
            "qualify_lead": "qualification",
            "close_conversion": "conversion"
        }
        return mapping.get(objective, "exploration")
    
    def _map_readiness_to_interest(self, readiness: float) -> str:
        """Mappe la disposition vers le niveau d'intérêt"""
        if readiness >= 0.7:
            return "high"
        elif readiness >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _get_next_recommended_action(self) -> str:
        """Recommande la prochaine action optimale"""
        
        if self.conversation_memory.is_ready_for_conversion():
            return "Organiser visite ou prise de contact"
        
        if self.conversation_memory.focused_property and len(self.conversation_memory.interest_signals) >= 1:
            return "Qualifier l'intérêt et proposer visite"
        
        if self.conversation_memory.properties_shown and not self.conversation_memory.focused_property:
            return "Encourager sélection d'une propriété"
        
        if self.conversation_memory.get_completion_score() < 0.5:
            return "Collecter plus de critères"
        
        if not self.conversation_memory.properties_shown:
            return "Effectuer recherche de propriétés"
        
        return "Continuer conversation naturelle"
    
    def _should_search_aggressively(self, user_message: str, memory: ObjectiveMemory) -> Dict[str, bool]:
        """Détermine de manière ultra-agressive si une recherche doit être faite"""
        
        message_lower = user_message.lower()
        
        # 1. Mots-clés de recherche explicites
        explicit_keywords = [
            'voir', 'montrer', 'proposer', 'options', 'disponible', 'propriétés',
            'appartements', 'résultats', 'chercher', 'trouver', 'qu\'avez-vous'
        ]
        
        # 2. Mentions de critères immobiliers
        property_keywords = [
            'appartement', 'villa', 'maison', 'studio', 'duplex',
            'chambres', 'pièces', 'surface', 'm²'
        ]
        
        # 3. Mentions de lieux
        location_keywords = [
            'anfa', 'maarif', 'gauthier', 'bourgogne', 'ain diab', 'corniche',
            'casablanca', 'rabat', 'marrakech', 'tanger', 'quartier', 'secteur'
        ]
        
        # 4. Mentions de budget
        budget_keywords = [
            'budget', 'prix', 'million', 'mad', 'dh', 'dirham', 'coût'
        ]
        
        # LOGIQUE ULTRA-AGRESSIVE
        should_search = False
        
        # Recherche explicite
        if any(keyword in message_lower for keyword in explicit_keywords):
            should_search = True
        
        # Critères immobiliers mentionnés
        elif any(keyword in message_lower for keyword in property_keywords):
            should_search = True
        
        # Localisation mentionnée
        elif any(keyword in message_lower for keyword in location_keywords):
            should_search = True
        
        # Budget mentionné
        elif any(keyword in message_lower for keyword in budget_keywords):
            should_search = True
        
        # Si on a déjà des critères minimaux en mémoire
        elif (memory.locations_preferred or memory.property_type or 
              memory.bedrooms_min or memory.budget_max):
            should_search = True
        
        # Phrases de recherche générales
        elif any(phrase in message_lower for phrase in [
            'je cherche', 'je veux', 'j\'aimerais', 'besoin de',
            'recherche', 'trouve', 'il y a', 'avez-vous'
        ]):
            should_search = True
        
        return {"should_search": should_search}
    
    def _build_broader_search_query(self, user_message: str, memory: ObjectiveMemory) -> str:
        """Construit une requête de recherche élargie si la première ne donne rien"""
        
        query_parts = []
        
        # Message utilisateur nettoyé
        cleaned_message = self._clean_message_for_search(user_message)
        if cleaned_message:
            query_parts.append(cleaned_message)
        
        # Seulement les critères les plus importants
        if memory.property_type:
            query_parts.append(memory.property_type)
        
        if memory.locations_preferred:
            # Prendre seulement la première location
            query_parts.append(memory.locations_preferred[0])
        
        # Pas de contraintes de budget pour élargir
        
        return " ".join(query_parts) if query_parts else "appartement casablanca"
    
    def _get_error_fallback(self, user_message: str) -> Dict[str, Any]:
        """Réponse de fallback en cas d'erreur"""
        
        return {
            "message": "Je m'excuse pour ce petit problème technique. Pouvez-vous me redire ce que vous recherchez ?",
            "phase": "exploration",
            "interest_level": "medium",
            "client_info": {
                "name": None, "phone": None, "email": None,
                "budget_min": None, "budget_max": None,
                "preferred_locations": [], "property_type": None,
                "bedrooms_min": None, "move_in_timeline": None,
                "financing_status": None, "specific_requirements": [],
                "viewing_preference": None
            },
            "search_results": None,
            "conversation_stats": {
                "message_count": self.message_count,
                "duration_minutes": 0,
                "objective": "discover_needs",
                "conversion_readiness": 0.0,
                "completion_score": 0.0
            },
            "error": True
        }
    
    # ========================================================================================
    # MÉTHODES UTILITAIRES POUR MONITORING ET OPTIMISATION
    # ========================================================================================
    
    def get_conversation_insights(self) -> Dict[str, Any]:
        """Insights détaillés pour optimisation"""
        
        return {
            "memory_state": {
                "completion_score": self.conversation_memory.get_completion_score(),
                "conversion_readiness": self.conversation_memory.conversion_readiness,
                "current_objective": self.conversation_memory.current_objective,
                "interest_signals": self.conversation_memory.interest_signals,
                "client_satisfaction": self.conversation_memory.client_satisfaction
            },
            "progression_analysis": {
                "messages_to_first_criteria": self._analyze_criteria_collection_speed(),
                "search_to_focus_efficiency": self._analyze_focus_efficiency(),
                "interest_signal_progression": self._analyze_interest_progression(),
                "conversion_bottlenecks": self._identify_conversion_bottlenecks()
            },
            "optimization_recommendations": {
                "next_action": self._get_next_recommended_action(),
                "conversation_adjustment": self._get_conversation_adjustment_advice(),
                "urgency_level": self._assess_conversation_urgency()
            }
        }
    
    def _analyze_criteria_collection_speed(self) -> int:
        """Analyse la vitesse de collecte des critères"""
        if self.conversation_memory.get_completion_score() > 0:
            return self.message_count
        return -1
    
    def _analyze_focus_efficiency(self) -> float:
        """Analyse l'efficacité du passage à la focalisation"""
        if self.conversation_memory.focused_property and self.conversation_memory.properties_shown:
            return len(self.conversation_memory.properties_shown) / max(1, self.message_count)
        return 0.0
    
    def _analyze_interest_progression(self) -> Dict[str, Any]:
        """Analyse la progression des signaux d'intérêt"""
        return {
            "signals_per_message": len(self.conversation_memory.interest_signals) / max(1, self.message_count),
            "signal_types": list(set(self.conversation_memory.interest_signals)),
            "progression_speed": "fast" if len(self.conversation_memory.interest_signals) >= self.message_count * 0.5 else "normal"
        }
    
    def _identify_conversion_bottlenecks(self) -> List[str]:
        """Identifie les blocages vers la conversion"""
        bottlenecks = []
        
        if self.message_count > 5 and self.conversation_memory.get_completion_score() < 0.4:
            bottlenecks.append("Collecte de critères lente")
        
        if len(self.conversation_memory.properties_shown) > 5 and not self.conversation_memory.focused_property:
            bottlenecks.append("Client ne sélectionne pas de propriété")
        
        if self.conversation_memory.focused_property and len(self.conversation_memory.interest_signals) < 2:
            bottlenecks.append("Signaux d'intérêt insuffisants")
        
        if self.conversation_memory.client_satisfaction == "frustrated":
            bottlenecks.append("Satisfaction client en baisse")
        
        if self.conversation_memory.conversion_readiness > 0.7 and not (self.conversation_memory.name and (self.conversation_memory.phone or self.conversation_memory.email)):
            bottlenecks.append("Informations de contact manquantes")
        
        return bottlenecks
    
    def _get_conversation_adjustment_advice(self) -> str:
        """Conseils d'ajustement de conversation"""
        
        if self.conversation_memory.client_satisfaction == "frustrated":
            return "Ralentir et rassurer le client"
        
        if self.conversation_memory.get_completion_score() < 0.3 and self.message_count > 3:
            return "Encourager plus activement le partage de critères"
        
        if len(self.conversation_memory.properties_shown) > 3 and not self.conversation_memory.focused_property:
            return "Guider vers la sélection d'une propriété"
        
        if self.conversation_memory.conversion_readiness > 0.6:
            return "Proposer actions concrètes (visite, contact)"
        
        return "Continuer naturellement"
    
    def _assess_conversation_urgency(self) -> str:
        """Évalue l'urgence de la conversation"""
        
        if self.conversation_memory.client_satisfaction == "frustrated":
            return "high"
        
        if self.conversation_memory.conversion_readiness > 0.8:
            return "high"
        
        if self.message_count > 8 and self.conversation_memory.get_completion_score() < 0.5:
            return "medium"
        
        return "low"
    
    def export_conversation_data(self) -> Dict[str, Any]:
        """Export complet des données de conversation"""
        
        return {
            "memory_snapshot": {
                "name": self.conversation_memory.name,
                "phone": self.conversation_memory.phone,
                "email": self.conversation_memory.email,
                "budget_max": self.conversation_memory.budget_max,
                "budget_min": self.conversation_memory.budget_min,
                "locations_preferred": self.conversation_memory.locations_preferred,
                "property_type": self.conversation_memory.property_type,
                "bedrooms_min": self.conversation_memory.bedrooms_min,
                "current_objective": self.conversation_memory.current_objective,
                "focused_property": self.conversation_memory.focused_property,
                "interest_signals": self.conversation_memory.interest_signals,
                "conversion_readiness": self.conversation_memory.conversion_readiness,
                "client_satisfaction": self.conversation_memory.client_satisfaction
            },
            "conversation_metrics": {
                "message_count": self.message_count,
                "duration_minutes": (datetime.now() - self.conversation_start).total_seconds() / 60,
                "completion_score": self.conversation_memory.get_completion_score(),
                "properties_shown_count": len(self.conversation_memory.properties_shown),
                "conversation_flow": self.conversation_memory.conversation_flow,
                "ready_for_conversion": self.conversation_memory.is_ready_for_conversion()
            },
            "conversation_history": self.conversation_history,
            "insights": self.get_conversation_insights()
        }
    
    def reset_conversation(self):
        """Remet à zéro la conversation"""
        self.conversation_memory = ObjectiveMemory()
        self.conversation_history = []
        self.message_count = 0
        self.conversation_start = datetime.now()
        logger.info("🔄 Conversation réinitialisée")

# ========================================================================================
# CLASSE DE COMPATIBILITÉ POUR MAINTENIR LES IMPORTS
# ========================================================================================

class ModularConversationManagerNatural(ModularConversationManagerEnhanced):
    """Alias pour compatibilité avec les anciens imports"""
    
    def __init__(self, groq_client, model: str = "llama3-70b-8192"):
        super().__init__(groq_client, model)
        logger.info("🔄 Interface de compatibilité 'Natural' initialisée")

# ========================================================================================
# EXPORT ET DOCUMENTATION
# ========================================================================================

__all__ = [
    # Classes principales
    'ObjectiveMemory',
    'IntentDetector', 
    'SmartInfoExtractor',
    'IntelligentSearchQueryBuilder',
    'ObjectiveResponseGenerator',
    
    # Gestionnaire principal
    'ModularConversationManagerEnhanced',
    
    # Compatibilité
    'ModularConversationManagerNatural'
]

"""
INSTRUCTIONS D'UTILISATION:

1. REMPLACEMENT DIRECT:
   # L'import reste exactement le même
   from intelligent_conversation_core import ModularConversationManagerEnhanced
   
   # Ou pour la compatibilité Natural
   from intelligent_conversation_core import ModularConversationManagerNatural

2. INITIALISATION IDENTIQUE:
   manager = ModularConversationManagerEnhanced(groq_client)

3. UTILISATION IDENTIQUE:
   result = await manager.process_message(user_message, search_function)

4. NOUVELLES FONCTIONNALITÉS:
   ✅ Orienté objectifs de conversion
   ✅ Collecte d'informations naturelle
   ✅ Détection intelligente d'intentions
   ✅ Progression automatique vers la qualification
   ✅ Signaux d'intérêt détectés automatiquement
   ✅ Focalisation intelligente sur propriétés
   ✅ Insights de conversion en temps réel

5. AVANTAGES IMMÉDIATS:
   ✅ Plus user-friendly (moins de questions)
   ✅ Progression naturelle vers l'objectif
   ✅ Détection automatique des opportunités
   ✅ Réponses orientées conversion
   ✅ Insights détaillés pour optimisation
   ✅ 100% compatible avec le code existant

6. MONITORING AVANCÉ:
   insights = manager.get_conversation_insights()
   readiness = manager.conversation_memory.conversion_readiness
   objective = manager.conversation_memory.current_objective
"""
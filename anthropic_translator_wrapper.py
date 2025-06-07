# ========================================================================================
# ANTHROPIC TRANSLATOR WRAPPER - VERSION COMPATIBLE AVEC L'ORIGINAL
# ========================================================================================

import asyncio
import json
import logging
import hashlib
import re
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from enum import Enum
import anthropic
import os

logger = logging.getLogger(__name__)

# ========================================================================================
# CONFIGURATION
# ========================================================================================

class SupportedLanguage(Enum):
    FRENCH = "fr"
    ENGLISH = "en" 
    SPANISH = "es"
    GERMAN = "de"
    DUTCH = "nl"
    ARABIC = "ar"
    DARIJA = "ary"

LANGUAGE_NAMES = {
    SupportedLanguage.FRENCH: "français",
    SupportedLanguage.ENGLISH: "anglais",
    SupportedLanguage.SPANISH: "espagnol", 
    SupportedLanguage.GERMAN: "allemand",
    SupportedLanguage.DUTCH: "néerlandais",
    SupportedLanguage.ARABIC: "arabe standard",
    SupportedLanguage.DARIJA: "darija marocain"
}

# Lieux marocains à préserver
PROTECTED_PLACES = {
    "maarif": ["maarif", "ma arif", "maârif"],
    "anfa": ["anfa", "anfa superieur", "anfa supérieur"],
    "centre_ville": ["centre ville", "centre-ville", "downtown"],
    "bourgogne": ["bourgogne"],
    "palmier": ["palmier", "les palmiers"],
    "val_fleuri": ["val fleuri", "val-fleuri"],
    "ain_diab": ["ain diab", "ain-diab", "corniche"],
    "finance_city": ["CFC", "casablanca finance city", "finance city"],
    "marina": ["Casa marina", "casablanca marina"],
    "racine": ["racine"],
    "california": ["california", "californie"],
    "oasis": ["oasis"],
    "cil": ["cil", "c.i.l", "cite d insertion locale", "cité d insertion locale"],
    "gauthier": ["gauthier"],
    "les_princesses": ["les princesses", "princesses"],
    "beausejour": ["beausejour", "beauséjour"],
    "derb_ghallef": ["derb ghallef"],
    "longchamp": ["longchamp"],
    "chantilly": ["chantilly"],
    "chantimar": ["chantimar"],
    "riviera": ["riviera"],
    "velodrome": ["velodrome", "vélodrome"],
    "ain_borja": ["ain borja", "aïn borja"],
    "midi": ["midi"],
    "sidi_maarouf": ["sidi maarouf", "sidi maârouf"],
    "oulfa": ["oulfa"],
    "lissasfa": ["lissasfa"],
    "hay_hassani": ["hay hassani"],
    "hay_salama": ["hay salama"],
    "hay_nassim": ["hay nassim"],
    "hay_el_hanaa": ["hay el hanna", "hay el hanaa", "el hanaa"],
    "ain_sebaa": ["ain sebaa", "aïn sebaâ", "ain sebaâ"],
    "sidi_bernoussi": ["sidi bernoussi"],
    "sidi_moumen": ["sidi moumen"],
    "salmia": ["salmia", "salmia 1", "salmia 2"],
    "moulay_rachid": ["moulay rachid"],
    "sidi_othmane": ["sidi othmane"],
    "hay_mohammadi": ["hay mohammadi"],
    "derb_sultan": ["derb sultan"],
    "mers_sultan": ["mers sultan"],
    "sidi_belyout": ["sidi belyout"],
    "sbata": ["sbata"],
    "ben_msik": ["ben msik", "ben m sik"],
    "ain_chock": ["ain chock", "aïn chok"],
    "hay_al_qods": ["hay al qods", "al qods"],
    "inara": ["inara"],
    "franceville": ["franceville"],
    "laimoune": ["laimoune", "laymoune"],
    "mandarona": ["mandarona"],
    "al_manar": ["al manar", "manar"],
    "florida": ["florida"],
    "attacharouk": ["attacharouk"],
    "alia": ["alia", "alia 1", "alia 2"],
    "plateau": ["plateau"],
    "zenata": ["zenata"],
    "les_roches_noires": ["roches noires", "les roches noires"],
}


# ========================================================================================
# WRAPPER TRADUCTEUR PRINCIPAL
# ========================================================================================

class AnthropicTranslatorWrapper:
    """Wrapper 100% LLM qui préserve votre code existant"""
    
    def __init__(self, anthropic_api_key: str = None, original_chatbot_manager=None):
        """
        anthropic_api_key: Votre clé API Anthropic
        original_chatbot_manager: Votre ModularConversationManager
        """
        if anthropic_api_key is None:
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            raise ValueError("❌ Clé API Anthropic requise")
            
        self.claude = anthropic.Anthropic(api_key=anthropic_api_key)
        self.original_chatbot = original_chatbot_manager
        self.session_languages = {}
        self.translation_cache = {}
        self.anthropic_api_key = anthropic_api_key
        
        logger.info("🤖 Wrapper traducteur Anthropic initialisé")
    
    async def process_user_message_with_translation(self, user_message: str, client_id: str) -> Dict[str, Any]:
        """
        POINT D'ENTRÉE PRINCIPAL - remplace chatbot.process_message()
        """
        try:
            logger.info(f"🌍 Message reçu: {user_message[:50]}...")
            
            # ÉTAPE 1+2 COMBINÉES: Détection + Traduction en 1 seul appel
            detected_language, french_message = await self._detect_and_translate_to_french(user_message)
            self.session_languages[client_id] = detected_language
            
            logger.info(f"✅ Langue détectée pour client {client_id}: {detected_language.value}")
            logger.info(f"📝 Message traduit en français pour chatbot: {french_message[:50]}...")
            
            # ÉTAPE 3: Exécution de votre code existant
            if not self.original_chatbot:
                raise ValueError("❌ Chatbot original non défini")
                
            logger.info("🚀 Exécution chatbot original...")
            original_response = await self.original_chatbot.process_message(french_message)
            logger.info(f"🤖 Réponse brute du chatbot original (fr): {json.dumps(original_response)[:200]}...")
            
            # ÉTAPE 4: Traduction de la réponse si nécessaire
            if detected_language == SupportedLanguage.FRENCH:
                final_response = original_response
                logger.info("✅ Langue française, pas de traduction de la réponse.")
            else:
                logger.info(f"🌍 Traduction réponse vers {LANGUAGE_NAMES[detected_language]} (depuis français)...")
                final_response = await self._translate_full_response(original_response, detected_language)
                logger.info(f"✅ Réponse traduite: {json.dumps(final_response)[:200]}...")
            
            # ÉTAPE 5: Métadonnées
            final_response["translation_metadata"] = {
                "detected_language": detected_language.value,
                "translation_performed": detected_language != SupportedLanguage.FRENCH,
                "original_message": user_message,
                "french_message": french_message,
                "cache_hits": len([k for k in self.translation_cache.keys() if user_message[:20] in k])
            }
            
            logger.info("✅ Traitement multilingue terminé")
            return final_response
            
        except Exception as e:
            logger.error(f"❌ Erreur wrapper: {str(e)}")
            # Fallback sécurisé
            if self.original_chatbot:
                logger.info("🔄 Fallback vers code original")
                # Note: Le fallback ne traduira pas le message si l'original est en français
                return await self.original_chatbot.process_message(user_message)
            else:
                raise e
    
    async def _detect_and_translate_to_french(self, text: str) -> Tuple[SupportedLanguage, str]:
        """OPTIMISATION: Détection + Traduction en 1 seul appel API"""
        
        # OPTIMISATION ULTRA: Détection heuristique française sans API
        if self._is_likely_french(text):
            logger.info("🚀 Français détecté heuristiquement (0 API call)")
            return SupportedLanguage.FRENCH, text
        
        # Cache combiné
        cache_key = f"detect_translate_{hashlib.md5(text.encode()).hexdigest()[:10]}"
        if cache_key in self.translation_cache:
            cached_result = self.translation_cache[cache_key]
            logger.info(f"🎯 Cache hit detect+translate: {cached_result['language']}")
            return SupportedLanguage(cached_result['language']), cached_result['french_text']
        
        prompt = f"""Analyse ce message et réponds avec un JSON contenant:
1. La langue détectée (fr/en/es/de/nl/ar/ary)
2. La traduction française (ou le texte original si déjà français)

Langues possibles:
- français: fr
- anglais: en  
- espagnol: es
- allemand: de
- néerlandais: nl
- arabe standard: ar
- darija marocain: ary

Message: "{text}"

Réponds UNIQUEMENT avec ce JSON:
{{"language": "CODE_LANGUE", "french_text": "TRADUCTION_FRANCAISE"}}"""

        try:
            response = self.claude.messages.create(
                model="claude-3-haiku-20240307",  # Modèle plus rapide !
                max_tokens=300,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result_text = response.content[0].text.strip()
            
            # Parse JSON
            if result_text.startswith('```json'):
                result_text = result_text[7:-3]
            elif result_text.startswith('```'):
                result_text = result_text[3:-3]
                
            result = json.loads(result_text)
            
            # Validation
            detected_code = result.get("language", "fr").lower()
            french_text = result.get("french_text", text)
            
            valid_codes = {lang.value: lang for lang in SupportedLanguage}
            detected_language = valid_codes.get(detected_code, SupportedLanguage.FRENCH)
            
            # Cache
            self.translation_cache[cache_key] = {
                "language": detected_language.value,
                "french_text": french_text
            }
            
            logger.info(f"🚀 Détecté + traduit en 1 appel: {detected_code}")
            return detected_language, french_text
            
        except Exception as e:
            logger.error(f"❌ Erreur detect+translate: {str(e)}")
            # Fallback vers méthode séparée
            detected_language = await self._detect_language_with_claude(text)
            if detected_language == SupportedLanguage.FRENCH:
                return detected_language, text
            else:
                french_text = await self._translate_to_french(text, detected_language)
                return detected_language, french_text
    
    def _is_likely_french(self, text: str) -> bool:
        """Détection heuristique française sans API"""
        french_indicators = [
            'bonjour', 'salut', 'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles',
            'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'dans', 'avec', 'pour',
            'cherche', 'recherche', 'appartement', 'maison', 'villa', 'chambre',
            'casablanca', 'rabat', 'marrakech', 'fes', 'agadir', 'tanger',
            'prix', 'budget', 'louer', 'acheter', 'vendre', 'vente', 'location',
            'merci', 'stp', 'svp', 'peux', 'peut', 'veux', 'veux', 'besoin'
        ]
        
        text_lower = text.lower()
        french_word_count = sum(1 for word in french_indicators if word in text_lower)
        
        # Si 2+ mots français détectés et pas de mots anglais évidents
        english_indicators = ['hello', 'hi', 'apartment', 'house', 'price', 'need', 'want', 'looking']
        english_word_count = sum(1 for word in english_indicators if word in text_lower)
        
        return french_word_count >= 2 and english_word_count == 0

    async def _detect_language_with_claude(self, text: str) -> SupportedLanguage:
        """Détection de langue 100% LLM"""
        
        # Cache pour éviter re-détections
        cache_key = f"detect_{hashlib.md5(text.encode()).hexdigest()[:10]}"
        if cache_key in self.translation_cache:
            cached_lang = self.translation_cache[cache_key]
            logger.info(f"🎯 Cache hit détection: {cached_lang}")
            return SupportedLanguage(cached_lang)
        
        prompt = f"""Tu es un expert en détection de langues pour l'immobilier au Maroc.

Analyse ce message et détermine la langue. Options possibles:
- français (fr)
- anglais (en) 
- espagnol (es)
- allemand (de)
- néerlandais (nl)
- arabe standard (ar)
- darija marocain (ary)

INDICES IMPORTANTS:
- Mots darija: "واخا", "بغيت", "كنقلب", "شحال", "فين", "كيفاش" = darija (ary)
- Arabe classique sans darija = arabe standard (ar)
- Focus sur contexte immobilier

Message: "{text}"

Réponds SEULEMENT avec le code (fr/en/es/de/nl/ar/ary):"""

        try:
            response = self.claude.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=10,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            detected_code = response.content[0].text.strip().lower()
            
            # Validation
            valid_codes = {lang.value: lang for lang in SupportedLanguage}
            detected_language = valid_codes.get(detected_code, SupportedLanguage.FRENCH)
            
            # Cache
            self.translation_cache[cache_key] = detected_language.value
            
            logger.info(f"🤖 Langue détectée: {detected_code} -> {LANGUAGE_NAMES[detected_language]}")
            return detected_language
            
        except Exception as e:
            logger.error(f"❌ Erreur détection: {str(e)}")
            return SupportedLanguage.FRENCH
    
    async def _robust_translate(self, prompt, max_retries=2, timeout=10):
        """Robust translation call with timeout and retries (compatible with sync Anthropic SDK)."""
        loop = asyncio.get_running_loop()
        for attempt in range(max_retries):
            try:
                response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self.claude.messages.create(
                            model="claude-3-5-sonnet-20240620",
                            max_tokens=1500,
                            temperature=0.1,
                            messages=[{"role": "user", "content": prompt}]
                        )
                    ),
                    timeout=timeout
                )
                translated_json = response.content[0].text.strip()
                logger.debug(f"🟡 Réponse brute Claude (avant parsing): {translated_json[:300]}")
                return translated_json
            except Exception as e:
                if "overloaded" in str(e).lower() or "timeout" in str(e).lower():
                    logger.warning(f"Retrying translation (attempt {attempt+1}) due to: {e}")
                    await asyncio.sleep(1.5 * (attempt + 1))
                else:
                    logger.error(f"Translation failed: {e}")
                    break
        logger.error("Translation failed after retries, returning original text.")
        return None

    async def _translate_to_french(self, text: str, source_language: SupportedLanguage) -> str:
        """Traduction robuste vers le français"""
        prompt = f"Traduis ce message en français. Réponds uniquement avec le texte traduit, sans explication.\nMessage: {text}"
        translated_text = await self._robust_translate(prompt)
        if not translated_text:
            logger.error("❌ Traduction vers français échouée, retour original.")
            return text
        return translated_text.strip()
    
    async def _translate_full_response(self, response_data: Dict[str, Any], target_language: SupportedLanguage) -> Dict[str, Any]:
        """Traduction robuste d'une réponse complète, traduisant tous les champs textuels pertinents."""
        translated_response = response_data.copy()
        logger.info(f"🌍 Traduction complète de la réponse vers {LANGUAGE_NAMES[target_language]}...")

        # --- Helper function to translate text fields within a property dictionary ---
        async def _translate_property_text_fields(prop_dict: Dict[str, Any], lang: SupportedLanguage) -> Dict[str, Any]:
            translated_prop = prop_dict.copy()
            fields_to_translate_in_prop = [
                "description", "neighborhood", "city", "notes", "details",
                "address", "amenities_description", "nearby_points_of_interest", # Add other potential text fields
            ]
            for field in fields_to_translate_in_prop:
                if field in translated_prop and isinstance(translated_prop[field], str) and translated_prop[field]:
                     translated_prop[field] = await self._translate_message(translated_prop[field], lang)
                     logger.debug(f"    - Traduit champ propriété '{field}'.")

            # Handle lists within property, e.g., list of amenities if they are strings
            property_list_fields = ["amenities_list", "features"] # Add other potential list fields
            for field in property_list_fields:
                 if field in translated_prop and isinstance(translated_prop[field], list):
                    translated_list = []
                    for item in translated_prop[field]:
                        if isinstance(item, str) and item:
                            translated_list.append(await self._translate_message(item, lang))
                        else:
                            translated_list.append(item) # Keep non-text elements
                    translated_prop[field] = translated_list
                    logger.debug(f"    - Traduit éléments de liste propriété '{field}'.")

            return translated_prop
        # --- End of helper function ---


        # --- Fields to translate at the root level ---

        # Translate the main 'message' field
        if "message" in translated_response and isinstance(translated_response["message"], str) and translated_response["message"]:
            translated_response["message"] = await self._translate_message(translated_response["message"], target_language)
            logger.debug("✅ Message principal traduit.")

        # Additional root-level string fields to translate
        root_fields_to_translate = [
            "suggested_response", # From ResponseStrategy
            "context_summary",    # Might be added to response
            "debug_info",         # Debugging text
            "closing_message",    # Specific closing message
            "context_acknowledgment", # Potential field
            "status_message",     # Potential status field
        ]
        for field in root_fields_to_translate:
             if field in translated_response and isinstance(translated_response[field], str) and translated_response[field]:
                  translated_response[field] = await self._translate_message(translated_response[field], target_language)
                  logger.debug(f"✅ Traduit champ racine '{field}'.")

        # --- Translate text within nested structures ---

        # Translate text fields within each property if search_results or focused_property exist
        # Handle search_results
        if "search_results" in translated_response and isinstance(translated_response["search_results"], dict) and "results" in translated_response["search_results"]:
            logger.info(f"🌍 Traduction des champs textuels de {len(translated_response['search_results']['results'])} propriétés (search_results)...")
            translated_results = []
            for prop in translated_response["search_results"]["results"]:
                translated_results.append(await _translate_property_text_fields(prop, target_language)) # Use helper
            translated_response["search_results"]["results"] = translated_results
            logger.info("✅ Champs textuels des propriétés (search_results) traduits.")

        # Handle focused_property (might have similar text fields)
        if "focused_property" in translated_response and isinstance(translated_response["focused_property"], dict):
             logger.info("🌍 Traduction des champs textuels de la propriété focalisée...")
             translated_response["focused_property"] = await _translate_property_text_fields(translated_response["focused_property"], target_language) # Use helper
             logger.info("✅ Champs textuels de la propriété focalisée traduits.")


        # Translate elements of lists of strings at the root level
        list_fields_to_translate = [
            "follow_up_questions", # From ResponseStrategy
            "missing_info",        # From ConversationAnalysis
            "next_questions",      # From ConversationAnalysis
            "notes",               # From ConversationState or others
            "recommendations",     # Potential field
            # "specific_requirements", # From ClientInfo (handled within client_info dict check below if needed)
            "property_highlights", # Potential field in property details list
        ]

        for field in list_fields_to_translate:
            if field in translated_response and isinstance(translated_response[field], list):
                translated_list = []
                logger.debug(f"🌍 Traduction des éléments de la liste '{field}'...")
                for item in translated_response[field]:
                    if isinstance(item, str) and item:
                        translated_list.append(await self._translate_message(item, target_language))
                    else:
                        translated_list.append(item) # Keep non-text elements as they are
                translated_response[field] = translated_list
                logger.debug(f"✅ Éléments de la liste '{field}' traduits.")

        # --- Handle ClientInfo fields if included (less common to show raw, but possible) ---
        # Check if client_info is a dictionary before iterating its fields.
        # Note: Specific requirements list within client_info is handled here.
        if "client_info" in translated_response and isinstance(translated_response["client_info"], dict):
            logger.debug("🌍 Checking client_info fields for translation...")
            client_info_string_fields = [
                "name", "phone", "email", "move_in_timeline",
                "financing_status", "viewing_preference",
            ]

            for field in client_info_string_fields:
                 if field in translated_response["client_info"] and isinstance(translated_response["client_info"][field], str) and translated_response["client_info"][field]:
                      translated_response["client_info"][field] = await self._translate_message(translated_response["client_info"][field], target_language)
                      logger.debug(f"  - Traduit champ client_info '{field}'.")

            # Handle specific_requirements list within client_info
            if "specific_requirements" in translated_response["client_info"] and isinstance(translated_response["client_info"]["specific_requirements"], list):
                 translated_list = []
                 logger.debug("🌍 Translating client_info specific_requirements list...")
                 for item in translated_response["client_info"]["specific_requirements"]:
                    if isinstance(item, str) and item:
                        translated_list.append(await self._translate_message(item, target_language))
                    else:
                        translated_list.append(item) # Keep non-text elements
                 translated_response["client_info"]["specific_requirements"] = translated_list
                 logger.debug("✅ client_info specific_requirements list translated.")



        # NOTE : Fields not explicitly handled here (e.g., deeply nested custom objects,
        #          non-string/list types) will not be translated.

        logger.info("✅ Traduction complète de la réponse terminée.")
        return translated_response
    
    async def _translate_message(self, message: str, target_language: SupportedLanguage) -> str:
        """Traduction ULTRA-RAPIDE d'un message simple"""
        
        # Cache spécialisé
        cache_key = f"msg_out_{target_language.value}_{hashlib.md5(message.encode()).hexdigest()[:8]}"
        if cache_key in self.translation_cache:
            logger.info(f"🎯 Cache hit message sortie")
            return self.translation_cache[cache_key]
        
        prompt = f"Traduis en {LANGUAGE_NAMES[target_language]}. Réponds SEULEMENT avec la traduction:\n\n{message}"
        
        try:
            response = self.claude.messages.create(
                model="claude-3-haiku-20240307",  # Modèle le plus rapide
                max_tokens=400,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            translated = response.content[0].text.strip()
            
            # Cache agressif
            self.translation_cache[cache_key] = translated
            logger.info(f"⚡ Message traduit ultra-rapide vers {target_language.value}")
            return translated
            
        except Exception as e:
            logger.error(f"❌ Erreur traduction ultra-rapide: {str(e)}")
            return message
    
    async def _translate_properties(self, properties: List[Dict], target_language: SupportedLanguage) -> List[Dict]:
        """DÉSACTIVÉ - Retourne les propriétés sans traduction pour performance"""
        logger.info(f"🚀 Propriétés non traduites (optimisation) - {len(properties)} propriétés")
        return properties  # Retour direct sans traduction
    
    def _protect_places(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Protège les noms de lieux marocains"""
        if not text:
            return text, {}
        
        placeholders = {}
        protected_text = text
        
        for i, place in enumerate(PROTECTED_PLACES):
            if place.lower() in text.lower():
                placeholder = f"__PLACE_{i}__"
                placeholders[placeholder] = place
                # Remplacement insensible à la casse
                pattern = re.compile(re.escape(place), re.IGNORECASE)
                protected_text = pattern.sub(placeholder, protected_text)
        
        return protected_text, placeholders
    
    def _restore_places(self, text: str, placeholders: Dict[str, str]) -> str:
        """Restaure les noms de lieux protégés"""
        if not placeholders:
            return text
        
        restored_text = text
        for placeholder, original_place in placeholders.items():
            restored_text = restored_text.replace(placeholder, original_place)
        
        return restored_text
    
    def get_session_language(self, client_id: str) -> Optional[SupportedLanguage]:
        """Récupère la langue de la session"""
        return self.session_languages.get(client_id)
    
    def clear_session(self, client_id: str):
        """Nettoie une session"""
        if client_id in self.session_languages:
            del self.session_languages[client_id]
            logger.info(f"🧹 Session nettoyée: {client_id}")
    
    async def translate_thinking_message(self, message: str, client_id: str) -> str:
        """Méthode publique pour traduire les messages de thinking"""
        session_language = self.session_languages.get(client_id)
        if not session_language or session_language == SupportedLanguage.FRENCH:
            return message
        
        return await self._translate_message(message, session_language)
    
    async def detect_language_for_client(self, message: str, client_id: str) -> SupportedLanguage:
        """Méthode publique pour détecter la langue d'un client"""
        detected_language = await self._detect_language_with_claude(message)
        self.session_languages[client_id] = detected_language
        return detected_language
    
    def get_translation_stats(self) -> Dict[str, Any]:
        """Statistiques de traduction"""
        languages_in_use = list(self.session_languages.values())
        language_counts = {}
        for lang in languages_in_use:
            lang_code = lang.value if hasattr(lang, "value") else str(lang)
            language_counts[lang_code] = language_counts.get(lang_code, 0) + 1
        
        return {
            "active_sessions": len(self.session_languages),
            "cache_size": len(self.translation_cache),
            "languages_distribution": language_counts,
            "supported_languages": [lang.value for lang in SupportedLanguage],
            "translation_engine": "anthropic_claude"
        }
    
    def clear_cache(self):
        """Vide le cache de traduction"""
        cache_size = len(self.translation_cache)
        self.translation_cache.clear()
        logger.info(f"🧹 Cache vidé: {cache_size} entrées supprimées")

# ========================================================================================
# UTILITAIRES DE TEST
# ========================================================================================

async def test_wrapper_simple():
    """Test rapide du wrapper"""
    import os
    
    # Test configuration
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ Définissez ANTHROPIC_API_KEY dans vos variables d'environnement")
        return
    
    # Mock chatbot simple
    class MockChatbot:
        async def process_message(self, message):
            return {
                "message": f"J'ai trouvé 3 propriétés à Casablanca pour votre demande: {message}",
                "search_results": {
                    "results": [
                        {
                            "description": "Bel appartement moderne avec vue mer",
                            "neighborhood": "Ain Diab", 
                            "city": "Casablanca",
                            "price": 2500000,
                            "area": 120,
                            "bedrooms": 3
                        }
                    ]
                }
            }
    
    # Test
    wrapper = AnthropicTranslatorWrapper(api_key, MockChatbot())
    
    test_messages = [
        ("Bonjour, je cherche un appartement", "test_fr"),
        ("Hello, I need an apartment", "test_en"),
        ("Hola, busco un apartamento", "test_es")
    ]
    
    for message, client_id in test_messages:
        print(f"\n🔸 Test: {message}")
        response = await wrapper.process_user_message_with_translation(message, client_id)
        print(f"🔸 Réponse: {response['message'][:80]}...")
        print(f"🔸 Langue: {response.get('translation_metadata', {}).get('detected_language', 'fr')}")
    
    print(f"\n📊 Stats: {wrapper.get_translation_stats()}")

if __name__ == "__main__":
    asyncio.run(test_wrapper_simple())
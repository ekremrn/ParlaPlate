"""
Single unified agent for ParlaPlate - handles both decision-making and conversation.
"""
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

from openai import OpenAI

from .schemas import MenuJSON, MenuItem, ChatTurn, Order, OrderItem, RestaurantProfile
from .personas import get_persona
from .prompts import UNIFIED_AGENT_SYSTEM, KEYWORD_EXTRACTION_SYSTEM
from .match import rank_candidates
from .utils import extract_json_from_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedWaitressAgent:
    """
    Single agent that handles both decision-making and user conversation.
    Enforces strict gating: no menu access until clear food intent is expressed.
    """
    
    def __init__(
        self,
        client: OpenAI,
        model_chat: str,
        model_vision: str,
        model_embed: str,
        restaurant: RestaurantProfile,
        menu: MenuJSON,
        persona_id: str
    ):
        """
        Initialize the unified waitress agent.
        
        Args:
            client: OpenAI client
            model_chat: Chat model name
            model_vision: Vision model name (kept for consistency)
            model_embed: Embedding model name
            restaurant: Restaurant profile
            menu: Menu data
            persona_id: Selected persona ID
        """
        self.client = client
        self.model_chat = model_chat
        self.model_vision = model_vision
        self.model_embed = model_embed
        self.restaurant = restaurant
        self.menu = menu
        self.persona = get_persona(persona_id)
        self.persona_id = persona_id
        
        logger.info(f"UnifiedWaitressAgent initialized with persona: {self.persona.name}")
    
    def extract_user_keywords(self, message: str) -> List[str]:
        """
        Extract keywords from user message.
        
        Args:
            message: User message
            
        Returns:
            List of extracted keywords
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_chat,
                messages=[
                    {"role": "system", "content": KEYWORD_EXTRACTION_SYSTEM},
                    {"role": "user", "content": message}
                ]
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON array
            json_text = extract_json_from_response(response_text)
            if not json_text:
                logger.warning(f"No JSON found in keyword extraction: {response_text[:50]}...")
                return []
            
            try:
                keywords = json.loads(json_text)
                if isinstance(keywords, list):
                    return [kw for kw in keywords if isinstance(kw, str)]
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in keyword extraction: {e} - JSON: {json_text}")
                return []
            
        except Exception as e:
            logger.error(f"Keyword extraction error: {e}")
        
        return []
    
    def lookup_candidates(self, keywords: List[str], constraints: Dict[str, Any]) -> List[MenuItem]:
        """
        Look up menu item candidates based on keywords and constraints.
        
        Args:
            keywords: Search keywords
            constraints: Filtering constraints
            
        Returns:
            List of candidate menu items
        """
        try:
            candidates = rank_candidates(
                self.client,
                self.model_embed,
                self.menu,
                keywords,
                constraints,
                top_k=8
            )
            
            logger.info(f"Found {len(candidates)} candidates for keywords: {keywords}")
            return candidates
            
        except Exception as e:
            logger.error(f"Candidate lookup error: {e}")
            return []
    
    def check_food_intent(self, history: List[ChatTurn], user_message: str) -> bool:
        """
        Check if user has expressed clear food intent in conversation.
        
        Args:
            history: Conversation history
            user_message: Current user message
            
        Returns:
            True if clear intent is detected
        """
        # Keywords that indicate food intent
        intent_keywords = [
            'acıktım', 'aç', 'yemek', 'yiyecek', 'lezzet', 'tat', 'menü',
            'et', 'tavuk', 'balık', 'sebze', 'salata', 'çorba', 'tatlı',
            'pizza', 'burger', 'pasta', 'döner', 'kebab', 'pilav',
            'kahvaltı', 'öğle', 'akşam', 'atıştırmalık',
            'vegetarian', 'vegan', 'gluten', 'spicy', 'mild', 'sweet',
            'salty', 'crispy', 'grilled', 'fried', 'soup', 'salad'
        ]
        
        # Delegation phrases
        delegation_phrases = [
            'sen seç', 'sana kalmış', 'öner', 'tavsiye', 'istediğin',
            'up to you', 'you choose', 'recommend', 'suggest'
        ]
        
        # Check current message and recent history
        all_text = user_message.lower()
        for turn in history[-3:]:  # Last 3 turns
            all_text += " " + turn.content.lower()
        
        # Check for delegation first
        if any(phrase in all_text for phrase in delegation_phrases):
            return True
        
        # Check for food intent keywords
        if any(keyword in all_text for keyword in intent_keywords):
            return True
        
        return False
    
    def build_system_prompt(self, candidates: Optional[List[MenuItem]] = None) -> str:
        """
        Build the system prompt for the unified agent.
        
        Args:
            candidates: Optional menu candidates for grounding
            
        Returns:
            Complete system prompt
        """
        # Restaurant context
        cuisine_str = ", ".join(self.restaurant.cuisine_tags) if self.restaurant.cuisine_tags else "diverse"
        service_str = ", ".join(self.restaurant.service_style) if self.restaurant.service_style else "casual"
        
        # Persona context
        persona_context = self.persona.system_prompt
        
        # Candidates JSON (if provided)
        candidates_json = ""
        if candidates:
            candidate_data = []
            for item in candidates[:3]:  # Top 3 only
                candidate_data.append({
                    "name": item.name,
                    "price": item.price,
                    "keywords": item.keywords[:5],
                    "allergens": item.allergens,
                    "ingredients": item.ingredients[:3]  # First 3 ingredients
                })
            candidates_json = json.dumps(candidate_data, ensure_ascii=False, indent=2)
        
        # Build full system prompt
        system_prompt = UNIFIED_AGENT_SYSTEM.format(
            service_style=service_str,
            cuisine_tags=cuisine_str,
            price_level=self.restaurant.price_level or "varies",
            summary_text=self.restaurant.summary_text,
            persona_context=persona_context,
            candidates_json=candidates_json
        )
        
        return system_prompt
    
    def parse_action_from_response(self, response_text: str) -> Tuple[Dict[str, Any], str]:
        """
        Parse action JSON and reply text from model response.
        
        Args:
            response_text: Raw model response
            
        Returns:
            Tuple of (action_dict, reply_text)
        """
        # Extract first JSON object (action)
        action_json = extract_json_from_response(response_text)
        action_dict = {"action": "ASK", "intent_clear": False, "notes": "fallback"}
        
        # Debug: log the raw response to understand what's happening
        logger.info(f"Raw API response: {response_text[:200]}...")
        
        if action_json:
            try:
                parsed_json = json.loads(action_json)
                if isinstance(parsed_json, list) and parsed_json:
                    # Look for the first dictionary in the list
                    dict_found = False
                    for item in parsed_json:
                        if isinstance(item, dict):
                            action_dict = item
                            dict_found = True
                            break
                    
                    if not dict_found:
                        # Handle the case where AI returns simple arrays like ['allergy_or_diet']
                        logger.warning(f"AI returned simple array instead of action dict: {parsed_json}")
                        # Convert simple array to meaningful action
                        if any(keyword in str(parsed_json).lower() for keyword in ['allergy', 'diet', 'preference']):
                            action_dict = {"action": "ASK", "intent_clear": False, "notes": "clarify_diet_allergy"}
                        elif any(keyword in str(parsed_json).lower() for keyword in ['drink', 'beverage']):
                            action_dict = {"action": "ASK", "intent_clear": False, "notes": "clarify_drink"}
                        elif any(keyword in str(parsed_json).lower() for keyword in ['side', 'accompaniment']):
                            action_dict = {"action": "ASK", "intent_clear": False, "notes": "clarify_side"}
                        else:
                            action_dict = {"action": "ASK", "intent_clear": False, "notes": "general_clarification"}
                elif isinstance(parsed_json, dict):
                    action_dict = parsed_json
                else:
                    logger.warning(f"Unexpected JSON format: {type(parsed_json)} - {parsed_json}")
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in action parsing: {e}")
        
        # Extract reply text (remove JSON completely)
        reply_text = response_text
        
        # Use regex to remove any JSON-like structures completely
        import re
        # Remove complete JSON objects (including incomplete ones)
        reply_text = re.sub(r'\{[^{}]*"action"[^{}]*\}', '', reply_text)
        # Remove incomplete JSON fragments that start with {"action"
        reply_text = re.sub(r'\{"action"[^}]*', '', reply_text)
        # Remove any leftover JSON fragments like ',"notes":"..."'
        reply_text = re.sub(r',"[^"]+"\s*:\s*"[^"]*"[^}]*', '', reply_text)
        reply_text = re.sub(r',"[^"]+"\s*:\s*[^,}]*', '', reply_text)
        
        # Clean up common prefixes and suffixes
        reply_text = reply_text.strip()
        while reply_text.startswith(('```', '\n', ' ', '\t', ',')):
            if reply_text.startswith('```'):
                reply_text = reply_text[3:].strip()
            elif reply_text.startswith('\n'):
                reply_text = reply_text[1:].strip()
            elif reply_text.startswith(','):
                reply_text = reply_text[1:].strip()
            else:
                reply_text = reply_text.lstrip()
        
        while reply_text.endswith(('```', '\n', ' ', '\t', '}')):
            if reply_text.endswith('```'):
                reply_text = reply_text[:-3].strip()
            elif reply_text.endswith('\n'):
                reply_text = reply_text[:-1].strip()
            elif reply_text.endswith('}'):
                reply_text = reply_text[:-1].strip()
            else:
                reply_text = reply_text.rstrip()
        
        # If reply is empty or just whitespace, use a fallback
        if not reply_text.strip():
            reply_text = "Nasıl yardımcı olabilirim?"
        
        
        return action_dict, reply_text
    
    def finalize_order(self, selected_items: List[str]) -> Order:
        """
        Create final order from selected item names.
        
        Args:
            selected_items: List of item names to order
            
        Returns:
            Order object
        """
        order_items = []
        for item_name in selected_items:
            order_items.append(OrderItem(name=item_name.strip(), notes=None))
        
        order = Order(
            order=order_items,
            persona=self.persona_id,
            restaurant=self.restaurant.display_name or self.restaurant.name or "Unknown",
            confidence=0.8,
            menu_json=f"opt/menu_content/{self.restaurant.name}.json" if self.restaurant.name else "unknown.json",
            timestamp=datetime.now()
        )
        
        logger.info(f"Finalized order with {len(order_items)} items")
        return order
    
    def respond(
        self, 
        history: List[ChatTurn], 
        user_message: str
    ) -> Tuple[str, str, List[MenuItem], Optional[Order]]:
        """
        Main entry point: single call that handles decision + reply.
        
        Args:
            history: Conversation history
            user_message: Current user message
            
        Returns:
            Tuple of (reply_text, action, candidates, maybe_order)
        """
        try:
            # Check if user has expressed clear food intent
            intent_clear = self.check_food_intent(history, user_message)
            
            # Build conversation context
            context_parts = []
            for turn in history[-5:]:  # Last 5 turns for context
                context_parts.append(f"{turn.role}: {turn.content}")
            context_parts.append(f"user: {user_message}")
            conversation_context = "\n".join(context_parts)
            
            # First call: get initial decision and reply
            system_prompt = self.build_system_prompt()
            
            response = self.client.chat.completions.create(
                model=self.model_chat,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": conversation_context}
                ]
            )
            
            response_text = response.choices[0].message.content.strip()
            action_dict, reply_text = self.parse_action_from_response(response_text)
            
            # Ensure action_dict is a dictionary
            if not isinstance(action_dict, dict):
                logger.error(f"action_dict is not a dict: {type(action_dict)} - {action_dict}")
                action_dict = {"action": "ASK", "intent_clear": False, "notes": "error_fallback"}
            
            action = action_dict.get("action", "ASK")
            candidates = []
            maybe_order = None
            
            logger.info(f"Unified agent decision: {action} (intent_clear: {intent_clear})")
            
            # If action requires menu access and intent is clear
            if action in ["LOOKUP", "RECOMMEND", "REFINE"] and intent_clear:
                # Extract keywords and get candidates
                keywords = self.extract_user_keywords(user_message)
                constraints = {
                    "diet": [],
                    "avoid_allergens": [],
                    "price_preference": None
                }
                
                candidates = self.lookup_candidates(keywords, constraints)
                
                # Second call: generate grounded response with candidates
                if candidates:
                    system_prompt_with_candidates = self.build_system_prompt(candidates)
                    
                    response = self.client.chat.completions.create(
                        model=self.model_chat,
                        messages=[
                            {"role": "system", "content": system_prompt_with_candidates},
                            {"role": "user", "content": f"{conversation_context}\n\nProvide specific recommendations from the menu candidates."}
                        ]
                    )
                    
                    response_text = response.choices[0].message.content.strip()
                    _, reply_text = self.parse_action_from_response(response_text)
            
            # Handle finalization
            elif action == "FINALIZE":
                # Extract item names from conversation (simplified)
                # In a real implementation, this would be more sophisticated
                selected_items = ["Example Item"]  # Placeholder
                maybe_order = self.finalize_order(selected_items)
                
                reply_text = "✅ Harika! Siparişiniz hazırlanıyor. Aşağıdan sipariş detaylarınızı indirebilirsiniz."
            
            return reply_text, action, candidates, maybe_order
            
        except Exception as e:
            logger.error(f"Unified agent error: {e}", exc_info=True)
            return "Özür dilerim, bir sorun oluştu. Lütfen tekrar deneyin.", "ASK", [], None
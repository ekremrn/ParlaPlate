"""
Streamlit UI for ParlaPlate - Restaurant Menu Ordering Assistant
"""
import os
import sys
import uuid
from typing import Dict
from datetime import datetime

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from parlaplate.schemas import ChatTurn, serialize_order
from parlaplate.personas import list_personas
from parlaplate.agent import UnifiedWaitressAgent
from parlaplate.utils import list_menu_jsons, load_menu_json
from parlaplate.match import load_or_compute_embeddings

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="ParlaPlate - AI Restoran Asistanı",
    page_icon="🍽️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.persona-card {
    border: 2px solid #f0f0f0;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s;
}
.persona-card:hover {
    border-color: #ff6b6b;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}
.persona-card.selected {
    border-color: #ff6b6b;
    background-color: #fff5f5;
}
.restaurant-card {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 12px;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)




def get_openai_config() -> Dict[str, str]:
    """Get OpenAI configuration from secrets or environment."""
    config = {
        "api_key": None,
        "model_chat": "gpt-4",
        "model_vision": "gpt-4-vision-preview", 
        "model_embed": "text-embedding-3-small",
    }
    
    # Try secrets first
    try:
        secrets = st.secrets.get("openai", {})
        if secrets:
            config.update({
                "api_key": secrets.get("key"),
                "model_chat": secrets.get("model_chat", config["model_chat"]),
                "model_vision": secrets.get("model_vision", config["model_vision"]),
                "model_embed": secrets.get("model_embed", config["model_embed"]),
            })
    except Exception:
        pass
    
    # Fallback to environment variables
    if not config.get("api_key"):
        config.update({
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model_chat": os.getenv("OPENAI_MODEL_CHAT", config["model_chat"]),
            "model_vision": os.getenv("OPENAI_MODEL_VISION", config["model_vision"]),
            "model_embed": os.getenv("OPENAI_MODEL_EMBED", config["model_embed"]),
        })
    
    return config


def initialize_session_state():
    """Initialize Streamlit session state."""
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
    
    if "reset_on_refresh" not in st.session_state:
        st.session_state.reset_on_refresh = True
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "selected_persona" not in st.session_state:
        st.session_state.selected_persona = None
    
    if "selected_restaurant" not in st.session_state:
        st.session_state.selected_restaurant = None
    
    if "menu_data" not in st.session_state:
        st.session_state.menu_data = None
    
    if "agent" not in st.session_state:
        st.session_state.agent = None
    
    if "order_finalized" not in st.session_state:
        st.session_state.order_finalized = False
    
    if "final_order" not in st.session_state:
        st.session_state.final_order = None


def reset_conversation():
    """Reset conversation state."""
    st.session_state.conversation_id = str(uuid.uuid4())
    st.session_state.chat_history = []
    st.session_state.order_finalized = False
    st.session_state.final_order = None
    st.session_state.agent = None


def setup_sidebar():
    """Setup sidebar with configuration."""
    st.sidebar.header("🔧 Yapılandırma")
    
    # OpenAI API Key input
    config = get_openai_config()
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        value=config.get("api_key", ""),
        type="password",
        help="OpenAI API anahtarınızı girin"
    )
    
    if api_key:
        config["api_key"] = api_key
    
    # Model configuration
    if config.get("api_key"):
        st.sidebar.text(f"Chat Model: {config['model_chat']}")
        st.sidebar.text(f"Vision Model: {config['model_vision']}")
        st.sidebar.text(f"Embedding Model: {config['model_embed']}")
    
    st.sidebar.divider()
    
    # Reset options
    st.session_state.reset_on_refresh = st.sidebar.checkbox(
        "Yenileme sırasında sohbeti sıfırla",
        value=st.session_state.get("reset_on_refresh", True)
    )
    
    if st.sidebar.button("🔄 Yeni Sohbet Başlat"):
        reset_conversation()
        st.rerun()
    
    return config


def show_persona_selection():
    """Show persona selection interface."""
    st.header("👤 Asistan Seçin")
    st.write("Size yardımcı olacak asistan tipini seçin:")
    
    personas = list_personas()
    
    cols = st.columns(2)
    for i, (persona_id, persona) in enumerate(personas.items()):
        col = cols[i % 2]
        
        with col:
            # Create persona card
            is_selected = st.session_state.selected_persona == persona_id
            
            if st.button(
                f"{persona.emoji} **{persona.name}**\n\n{persona.summary}",
                key=f"persona_{persona_id}",
                use_container_width=True
            ):
                st.session_state.selected_persona = persona_id
                st.rerun()


def show_restaurant_selection():
    """Show restaurant selection interface."""
    st.header("🏪 Restoran Seçin")
    
    menu_files = list_menu_jsons()
    
    if not menu_files:
        st.warning("Henüz menü bulunamadı. Lütfen PDF menüleri `opt/menu/` klasörüne yerleştirin ve CLI tool ile işleyin.")
        st.code("python -m tasks.menu_extract")
        return
    
    restaurant_options = []
    restaurant_data = {}
    
    for menu_file in menu_files:
        try:
            menu_json = load_menu_json(menu_file)
            restaurant_name = menu_json.restaurant.display_name or menu_json.restaurant.name or "Unknown"
            restaurant_options.append(restaurant_name)
            restaurant_data[restaurant_name] = {
                "file": menu_file,
                "menu": menu_json
            }
        except Exception as e:
            st.error(f"Menü yüklenemedi {menu_file}: {e}")
            continue
    
    if not restaurant_options:
        st.error("Geçerli menü bulunamadı.")
        return
    
    # Restaurant selection
    current_index = 0
    if st.session_state.selected_restaurant and st.session_state.selected_restaurant in restaurant_options:
        current_index = restaurant_options.index(st.session_state.selected_restaurant)
    
    selected_restaurant = st.selectbox(
        "Restoran seçin:",
        options=restaurant_options,
        index=current_index,
        key="restaurant_selector"
    )
    
    if selected_restaurant != st.session_state.selected_restaurant:
        st.session_state.selected_restaurant = selected_restaurant
        st.session_state.menu_data = restaurant_data[selected_restaurant]["menu"]
        # Reset conversation when restaurant changes
        st.session_state.chat_history = []
        st.session_state.agent = None
        st.session_state.order_finalized = False
        st.session_state.final_order = None
        st.rerun()
    
    # Show restaurant info
    if selected_restaurant:
        menu_data = restaurant_data[selected_restaurant]["menu"]
        restaurant = menu_data.restaurant
        
        st.subheader(f"📋 {restaurant.display_name or restaurant.name}")
        
        col1, col2 = st.columns(2)
        with col1:
            if restaurant.cuisine_tags:
                st.write(f"🍽️ **Mutfak:** {', '.join(restaurant.cuisine_tags)}")
            if restaurant.price_level:
                st.write(f"💰 **Fiyat Seviyesi:** {restaurant.price_level.title()}")
        
        with col2:
            st.write(f"📊 **Toplam Ürün:** {len(menu_data.items)}")
            if restaurant.popular_categories:
                st.write(f"🏷️ **Popüler Kategoriler:** {', '.join(restaurant.popular_categories[:3])}")
        
        if restaurant.summary_text:
            st.write(f"📝 {restaurant.summary_text}")
        
        # Show sample menu items
        with st.expander("Örnek Menü Öğeleri"):
            sample_items = menu_data.items[:5]
            for item in sample_items:
                price_str = f" - {item.price}" if item.price else ""
                category_str = f" ({item.category})" if item.category else ""
                st.write(f"• **{item.name}**{price_str}{category_str}")
        
        return menu_data
    
    return None


def get_welcome_message(persona, restaurant_name: str) -> str:
    """Generate persona-specific welcome message."""
    welcome_messages = {
        "ayla": f"Merhaba! Ben Ayla {persona.emoji} {restaurant_name}'a hoş geldiniz! Size sağlıklı ve dengeli seçenekler önerebilirim. Hangi tür yemekleri tercih ediyorsunuz?",
        
        "zeyna": f"Selam! Ben Zeyna {persona.emoji} {restaurant_name}'dayız! Bugün nasıl hissediyorsun? Ruh halina göre mükemmel lezzetler bulabilirim.",
        
        "mert": f"Merhaba! Ben Mert {persona.emoji} {restaurant_name}'a hoş geldin! En iyi fiyat-performans oranına sahip lezzetli seçenekleri bilirim. Bütçen ne kadar?",
        
        "alessandro": f"Buongiorno! Ben Alessandro {persona.emoji} {restaurant_name}'da size hizmet etmekten mutluluk duyarım. Yıllarca deneyimimle en lezzetli önerileri yapabilirim.",
        
        "lara": f"Merhaba canım! Ben Lara {persona.emoji} {restaurant_name}'a hoş geldin! Seni rahatlatacak ve mutlu edecek lezzetler bulalım birlikte. Ne tür yemekler seviyorsun?"
    }
    
    persona_id = persona.name.lower()
    return welcome_messages.get(persona_id, f"Merhaba! {restaurant_name}'a hoş geldiniz! Size nasıl yardımcı olabilirim?")


def initialize_agent(config: Dict[str, str], menu_data, persona_id: str):
    """Initialize the unified waitress agent."""
    if not config.get("api_key"):
        st.error("OpenAI API key gerekli!")
        return None
    
    try:
        client = OpenAI(api_key=config["api_key"])
        
        agent = UnifiedWaitressAgent(
            client=client,
            model_chat=config["model_chat"],
            model_vision=config["model_vision"],
            model_embed=config["model_embed"],
            restaurant=menu_data.restaurant,
            menu=menu_data,
            persona_id=persona_id
        )
        
        # Pre-compute embeddings
        with st.spinner("Menü verileri hazırlanıyor..."):
            load_or_compute_embeddings(
                client, config["model_embed"], menu_data
            )
        
        return agent
        
    except Exception as e:
        st.error(f"Agent başlatılamadı: {e}")
        return None


def show_chat_interface(agent: UnifiedWaitressAgent, config: Dict[str, str]):
    """Show chat interface."""
    st.header("💬 Sohbet")
    
    # Show welcome message if no chat history
    if not st.session_state.chat_history:
        welcome_message = get_welcome_message(agent.persona, st.session_state.selected_restaurant)
        with st.chat_message("assistant"):
            st.write(welcome_message)
    
    # Display chat history
    for turn in st.session_state.chat_history:
        with st.chat_message(turn.role):
            st.write(turn.content)
    
    # Show order finalized message if applicable
    if st.session_state.order_finalized and st.session_state.final_order:
        st.success("✅ Siparişiniz alındı!")
        
        # Show order details
        st.subheader("📋 Sipariş Özeti")
        for item in st.session_state.final_order.order:
            notes_str = f" ({item.notes})" if item.notes else ""
            st.write(f"• {item.name}{notes_str}")
        
        # Download button
        order_json = serialize_order(st.session_state.final_order)
        st.download_button(
            "📥 Sipariş JSON İndir",
            data=order_json,
            file_name=f"order_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        return
    
    # Chat input (disabled if order is finalized)
    if not st.session_state.order_finalized:
        if prompt := st.chat_input("Mesajınızı yazın..."):
            # Add user message to history
            user_turn = ChatTurn(role="user", content=prompt)
            st.session_state.chat_history.append(user_turn)
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Get agent response
            with st.chat_message("assistant"):
                with st.spinner("Yanıt hazırlanıyor..."):
                    try:
                        # Use unified agent's respond method
                        response, action, candidates, maybe_order = agent.respond(
                            st.session_state.chat_history[:-1], prompt
                        )
                        
                        # Add assistant response to history
                        assistant_turn = ChatTurn(role="assistant", content=response)
                        st.session_state.chat_history.append(assistant_turn)
                        
                        st.write(response)
                        
                        # Handle order finalization
                        if action == "FINALIZE" and maybe_order:
                            st.session_state.final_order = maybe_order
                            st.session_state.order_finalized = True
                            
                        
                    except Exception as e:
                        error_msg = f"Bir hata oluştu: {e}"
                        st.error(error_msg)
                        
                        assistant_turn = ChatTurn(role="assistant", content=error_msg)
                        st.session_state.chat_history.append(assistant_turn)
            
            st.rerun()




def main():
    """Main application."""
    st.title("🍽️ ParlaPlate")
    st.subheader("AI Destekli Restoran Sipariş Asistanı")
    
    # Initialize session state
    initialize_session_state()
    
    # Setup sidebar and get config
    config = setup_sidebar()
    
    # Reset conversation on refresh if enabled
    if st.session_state.reset_on_refresh and len(st.session_state.chat_history) == 0:
        reset_conversation()
    
    # Step 1: Persona Selection
    if not st.session_state.selected_persona:
        show_persona_selection()
        return
    
    # Step 2: Restaurant Selection
    if not st.session_state.selected_restaurant:
        menu_data = show_restaurant_selection()
        if menu_data:
            st.session_state.menu_data = menu_data
        return
    
    # Step 3: Initialize Agent (if not already done)
    if not st.session_state.agent and st.session_state.menu_data:
        with st.spinner("AI asistan hazırlanıyor..."):
            agent = initialize_agent(
                config, 
                st.session_state.menu_data, 
                st.session_state.selected_persona
            )
            if agent:
                st.session_state.agent = agent
            else:
                return
    
    # Step 4: Chat Interface
    if st.session_state.agent:
        show_chat_interface(st.session_state.agent, config)
    
    # Show current selections in sidebar
    st.sidebar.divider()
    st.sidebar.subheader("🎯 Mevcut Seçimler")
    if st.session_state.selected_persona:
        personas = list_personas()
        persona = personas[st.session_state.selected_persona]
        st.sidebar.write(f"👤 **Asistan:** {persona.emoji} {persona.name}")
    
    if st.session_state.selected_restaurant:
        st.sidebar.write(f"🏪 **Restoran:** {st.session_state.selected_restaurant}")


if __name__ == "__main__":
    main()
"""
Pydantic schemas for ParlaPlate application.
"""
from datetime import datetime
from typing import List, Literal, Optional
import json

from pydantic import BaseModel, Field


class MenuItem(BaseModel):
    """Represents a single menu item with all its properties."""
    name: str = Field(..., description="Name of the menu item")
    price: Optional[str] = Field(None, description="Price as a string (e.g., '15.99 TL')")
    ingredients: List[str] = Field(default_factory=list, description="List of ingredients")
    keywords: List[str] = Field(default_factory=list, description="Keywords for search/matching")
    allergens: List[str] = Field(default_factory=list, description="Known allergens")
    category: Optional[str] = Field(None, description="Menu category (appetizers, mains, etc.)")
    spice_level: Optional[Literal["low", "medium", "high"]] = Field(None, description="Spice level")


class RestaurantProfile(BaseModel):
    """Restaurant profile information extracted from menu analysis."""
    name: Optional[str] = Field(None, description="Restaurant name")
    display_name: Optional[str] = Field(None, description="Display name for UI")
    cuisine_tags: List[str] = Field(default_factory=list, description="Cuisine type tags")
    price_level: Optional[Literal["low", "medium", "high"]] = Field(None, description="Overall price level")
    service_style: List[str] = Field(default_factory=list, description="Service style tags")
    diet_coverage: List[str] = Field(default_factory=list, description="Diet types supported")
    popular_categories: List[str] = Field(default_factory=list, description="Most popular menu categories")
    summary_text: str = Field(..., description="25-40 word summary of the restaurant")


class MenuJSON(BaseModel):
    """Complete menu data structure with restaurant profile and items."""
    restaurant: RestaurantProfile = Field(..., description="Restaurant profile")
    items: List[MenuItem] = Field(..., description="List of all menu items")


class OrderItem(BaseModel):
    """Individual item in an order."""
    name: str = Field(..., description="Name of the ordered item")
    notes: Optional[str] = Field(None, description="Special notes or modifications")


class Order(BaseModel):
    """Complete order information."""
    order: List[OrderItem] = Field(..., description="List of ordered items")
    persona: str = Field(..., description="Persona ID used for the order")
    restaurant: str = Field(..., description="Restaurant name")
    confidence: float = Field(..., description="Confidence level (0.0-1.0)")
    menu_json: str = Field(..., description="Path to menu JSON file")
    timestamp: datetime = Field(default_factory=datetime.now, description="Order timestamp")


class ChatTurn(BaseModel):
    """Individual chat message."""
    role: Literal["user", "assistant"] = Field(..., description="Message role")
    content: str = Field(..., description="Message content")




def serialize_order(order: Order) -> str:
    """
    Serialize an Order object to pretty JSON string.
    
    Args:
        order: Order object to serialize
        
    Returns:
        Pretty formatted JSON string
    """
    return json.dumps(
        order.model_dump(mode="json"),
        indent=2,
        ensure_ascii=False,
        default=str
    )
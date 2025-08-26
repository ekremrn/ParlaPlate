"""
PDF menu extraction using GPT Vision API.
"""
import os
import json
import logging
import io
from typing import Tuple, List, Dict, Any
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image
from openai import OpenAI

from .schemas import MenuJSON, MenuItem, RestaurantProfile
from .utils import pil_to_base64_png, clean_filename, extract_json_from_response, merge_menu_items, save_menu_json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def render_pdf_page_to_pil(pdf_page: fitz.Page, dpi: int = 200) -> Image.Image:
    """
    Render a PDF page to PIL Image.
    
    Args:
        pdf_page: PyMuPDF page object
        dpi: Rendering DPI
        
    Returns:
        PIL Image of the page
    """
    # Render page to pixmap
    mat = fitz.Matrix(dpi / 72, dpi / 72)  # Scale factor for DPI
    pix = pdf_page.get_pixmap(matrix=mat)
    
    # Convert to PIL Image
    img_data = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_data))
    
    # Optionally resize if too large
    max_dimension = 1024
    if max(img.width, img.height) > max_dimension:
        ratio = max_dimension / max(img.width, img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    return img


def extract_items_from_page(
    client: OpenAI,
    model_vision: str,
    page_image: Image.Image,
    extraction_system: str,
    extraction_user: str,
    page_num: int
) -> List[Dict[str, Any]]:
    """
    Extract menu items from a single page image using Vision API.
    
    Args:
        client: OpenAI client
        model_vision: Vision model name
        page_image: PIL Image of the page
        extraction_system: System prompt for extraction
        extraction_user: User prompt for extraction
        page_num: Page number (for logging)
        
    Returns:
        List of menu item dictionaries
    """
    try:
        # Convert image to base64
        image_data_url = pil_to_base64_png(page_image)
        
        # Make API call
        response = client.chat.completions.create(
            model=model_vision,
            messages=[
                {"role": "system", "content": extraction_system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": extraction_user},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url}
                        }
                    ]
                }
            ]
        )
        
        response_text = response.choices[0].message.content.strip()
        logger.info(f"Page {page_num} response: {response_text[:100]}...")
        
        # Extract JSON from response
        json_text = extract_json_from_response(response_text)
        if not json_text:
            logger.warning(f"Page {page_num}: No JSON found in response")
            logger.debug(f"Full response: {response_text}")
            return []
        
        logger.info(f"Page {page_num}: Extracted JSON: {json_text[:200]}...")
        
        # Parse JSON
        items = json.loads(json_text)
        
        # Validate it's a list
        if not isinstance(items, list):
            logger.warning(f"Page {page_num}: Response is not a JSON array, got: {type(items)}")
            return []
        
        logger.info(f"Page {page_num}: Extracted {len(items)} items")
        return items
        
    except json.JSONDecodeError as e:
        logger.error(f"Page {page_num}: JSON decode error: {e}")
        return []
    except Exception as e:
        logger.error(f"Page {page_num}: Extraction error: {e}")
        return []


def create_restaurant_profile(
    client: OpenAI,
    model_chat: str,
    merged_items: List[Dict[str, Any]],
    summary_system: str,
    pdf_name: str
) -> RestaurantProfile:
    """
    Create restaurant profile from merged menu items.
    
    Args:
        client: OpenAI client
        model_chat: Chat model name
        merged_items: List of all menu items
        summary_system: System prompt for profiling
        pdf_name: Original PDF name
        
    Returns:
        RestaurantProfile object
    """
    try:
        # Prepare items for analysis
        items_json = json.dumps(merged_items, ensure_ascii=False, indent=2)
        
        response = client.chat.completions.create(
            model=model_chat,
            messages=[
                {"role": "system", "content": summary_system},
                {"role": "user", "content": f"Menu items:\n{items_json}"}
            ]
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Extract JSON from response
        json_text = extract_json_from_response(response_text)
        if not json_text:
            raise ValueError("No JSON found in restaurant profile response")
        
        profile_data = json.loads(json_text)
        
        # Ensure profile_data is a dictionary
        if isinstance(profile_data, list):
            # If it's a list, take the first element if it's a dict
            if profile_data and isinstance(profile_data[0], dict):
                profile_data = profile_data[0]
            else:
                # If list is empty or contains non-dict, create default
                profile_data = {}
        elif not isinstance(profile_data, dict):
            # If it's neither list nor dict, create default
            profile_data = {}
        
        # Use PDF name as fallback for restaurant name
        if not profile_data.get("name"):
            profile_data["name"] = clean_filename(pdf_name)
        if not profile_data.get("display_name"):
            profile_data["display_name"] = profile_data["name"].replace("_", " ").title()
        if not profile_data.get("summary_text"):
            # Create a more meaningful fallback summary based on menu analysis
            categories = set(item.get('category', 'Unknown') for item in merged_items if item.get('category'))
            price_range = "varies"
            
            # Analyze price range if available
            prices = []
            for item in merged_items:
                price_str = item.get('price', '')
                if price_str and '₺' in price_str:
                    # Extract first number from price string
                    import re
                    numbers = re.findall(r'\d+', price_str)
                    if numbers:
                        prices.append(int(numbers[0]))
            
            if prices:
                avg_price = sum(prices) / len(prices)
                if avg_price < 100:
                    price_range = "budget-friendly"
                elif avg_price < 300:
                    price_range = "moderate"
                else:
                    price_range = "premium"
            
            restaurant_name = profile_data.get("display_name", pdf_name.replace(".pdf", ""))
            category_list = list(categories)[:3]  # Top 3 categories
            category_text = ", ".join(category_list) if category_list else "diverse cuisine"
            
            profile_data["summary_text"] = f"{restaurant_name} offers {category_text} with a {price_range} price range. Features {len(merged_items)} menu items including popular options across different categories."
        
        return RestaurantProfile.model_validate(profile_data)
        
    except Exception as e:
        logger.error(f"Restaurant profile creation error: {e}")
        # Return default profile
        clean_name = clean_filename(pdf_name)
        return RestaurantProfile(
            name=clean_name,
            display_name=clean_name.replace("_", " ").title(),
            summary_text=f"Restaurant extracted from {pdf_name} with {len(merged_items)} menu items."
        )


def extract_menu_from_pdf_bytes(
    pdf_bytes: bytes,
    pdf_name: str,
    client: OpenAI,
    model_vision: str,
    model_chat: str,
    extraction_system: str,
    extraction_user: str,
    summary_system: str
) -> Tuple[MenuJSON, str]:
    """
    Extract menu from PDF bytes.
    
    Args:
        pdf_bytes: PDF file bytes
        pdf_name: PDF filename
        client: OpenAI client
        model_vision: Vision model name
        model_chat: Chat model name
        extraction_system: System prompt for extraction
        extraction_user: User prompt for extraction
        summary_system: System prompt for restaurant profiling
        
    Returns:
        Tuple of (MenuJSON object, output file path)
    """
    logger.info(f"Processing PDF: {pdf_name}")
    
    # Open PDF
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    all_page_items = []
    empty_pages = 0
    
    # Process each page
    for page_num in range(len(pdf_doc)):
        logger.info(f"Processing page {page_num + 1}/{len(pdf_doc)}")
        
        page = pdf_doc[page_num]
        
        # Render page to image
        page_image = render_pdf_page_to_pil(page)
        
        # Extract items from page
        page_items = extract_items_from_page(
            client, model_vision, page_image, 
            extraction_system, extraction_user, page_num + 1
        )
        
        if not page_items:
            empty_pages += 1
        else:
            all_page_items.append(page_items)
    
    logger.info(f"Processed {len(pdf_doc)} pages, {empty_pages} empty pages")

    # Close PDF document
    pdf_doc.close()
    
    # Merge items from all pages
    merged_items = merge_menu_items(all_page_items)
    logger.info(f"Total unique items: {len(merged_items)}")
    
    # Create MenuItem objects
    menu_items = []
    for item_dict in merged_items:
        try:
            menu_item = MenuItem.model_validate(item_dict)
            menu_items.append(menu_item)
        except Exception as e:
            logger.warning(f"Skipping invalid item: {e}")
            continue
    
    # Create restaurant profile
    restaurant_profile = create_restaurant_profile(
        client, model_chat, merged_items, summary_system, pdf_name
    )
    
    # Create MenuJSON
    menu_json = MenuJSON(
        restaurant=restaurant_profile,
        items=menu_items
    )
    
    # Save to output directory
    clean_name = clean_filename(pdf_name)
    output_path = f"opt/menu_content/{clean_name}.json"
    save_menu_json(menu_json, output_path)
    
    logger.info(f"Saved menu to: {output_path}")
    
    return menu_json, output_path


def extract_menu_from_pdf_path(
    pdf_path: str,
    client: OpenAI,
    model_vision: str,
    model_chat: str,
    extraction_system: str,
    extraction_user: str,
    summary_system: str
) -> Tuple[MenuJSON, str]:
    """
    Extract menu from PDF file path.
    
    Args:
        pdf_path: Path to PDF file
        client: OpenAI client
        model_vision: Vision model name
        model_chat: Chat model name
        extraction_system: System prompt for extraction
        extraction_user: User prompt for extraction
        summary_system: System prompt for restaurant profiling
        
    Returns:
        Tuple of (MenuJSON object, output file path)
    """
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()
    
    pdf_name = Path(pdf_path).name
    
    return extract_menu_from_pdf_bytes(
        pdf_bytes, pdf_name, client, model_vision, model_chat,
        extraction_system, extraction_user, summary_system
    )
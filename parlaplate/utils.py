"""
Utility functions for ParlaPlate.
"""
import os
import json
import hashlib
import re
import base64
from io import BytesIO
from typing import List, Optional
from pathlib import Path

from PIL import Image

from .schemas import MenuJSON


def list_menu_jsons(dir_path: str = "opt/menu_content") -> List[str]:
    """
    List all menu JSON files in a directory.
    
    Args:
        dir_path: Directory to search
        
    Returns:
        List of JSON file paths
    """
    if not os.path.exists(dir_path):
        return []
    
    json_files = []
    for filename in os.listdir(dir_path):
        if filename.endswith('.json') and not filename.endswith('.emb.npy'):
            json_files.append(os.path.join(dir_path, filename))
    
    return sorted(json_files)


def load_menu_json(path: str) -> MenuJSON:
    """
    Load menu JSON from file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        MenuJSON object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
        ValueError: If JSON doesn't match schema
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return MenuJSON.model_validate(data)


def save_menu_json(menu_json: MenuJSON, path: str) -> None:
    """
    Save menu JSON to file.
    
    Args:
        menu_json: MenuJSON object to save
        path: Output file path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(
            menu_json.model_dump(mode="json"),
            f,
            ensure_ascii=False,
            indent=2
        )


def sha256_text(text: str) -> str:
    """
    Compute SHA256 hash of text.
    
    Args:
        text: Input text
        
    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def price_bucket(price_str: Optional[str]) -> Optional[str]:
    """
    Categorize price string into low/medium/high buckets.
    
    Args:
        price_str: Price string (e.g., "15.99 TL", "$12.50")
        
    Returns:
        Price bucket or None if price can't be parsed
    """
    if not price_str:
        return None
    
    # Extract numeric value from price string
    numbers = re.findall(r'\d+(?:\.\d+)?', price_str)
    if not numbers:
        return None
    
    try:
        price_value = float(numbers[0])
        
        # Simple heuristic based on common price ranges
        # These thresholds might need adjustment based on currency and context
        if price_value < 20:
            return "low"
        elif price_value < 50:
            return "medium"
        else:
            return "high"
            
    except ValueError:
        return None


def pil_to_base64_png(img: Image.Image) -> str:
    """
    Convert PIL Image to base64-encoded PNG string.
    
    Args:
        img: PIL Image object
        
    Returns:
        Base64-encoded PNG data URL
    """
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"


def clean_filename(filename: str) -> str:
    """
    Clean filename for safe filesystem usage.
    
    Args:
        filename: Original filename
        
    Returns:
        Cleaned filename
    """
    # Remove extension and clean
    name = Path(filename).stem
    # Keep only alphanumeric chars, hyphens, and underscores
    cleaned = re.sub(r'[^a-zA-Z0-9\-_]', '_', name)
    # Remove multiple consecutive underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')
    
    return cleaned.lower() if cleaned else "menu"


def extract_json_from_response(response_text: str) -> Optional[str]:
    """
    Extract JSON from model response that might contain extra text.
    
    Args:
        response_text: Raw response text
        
    Returns:
        Extracted JSON string or None if not found
    """
    # Clean the response text
    response_text = response_text.strip()
    
    # First try to find JSON arrays (for menu extraction)
    array_start = -1
    bracket_count = 0
    
    for i, char in enumerate(response_text):
        if char == '[':
            if array_start == -1:
                array_start = i
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
            if bracket_count == 0 and array_start != -1:
                # Found complete JSON array
                json_str = response_text[array_start:i+1]
                try:
                    # Test if it's valid JSON
                    json.loads(json_str)
                    return json_str
                except:
                    # Reset and continue searching
                    array_start = -1
                    bracket_count = 0
                    continue
    
    # Then try to find JSON objects
    obj_start = -1
    brace_count = 0
    
    for i, char in enumerate(response_text):
        if char == '{':
            if obj_start == -1:
                obj_start = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and obj_start != -1:
                # Found complete JSON object
                json_str = response_text[obj_start:i+1]
                try:
                    # Test if it's valid JSON
                    json.loads(json_str)
                    return json_str
                except:
                    # Reset and continue searching
                    obj_start = -1
                    brace_count = 0
                    continue
    
    # Fallback to regex patterns with better matching
    json_patterns = [
        r'\[\s*\{.*?\}\s*\]',  # Array of objects
        r'\[.*?\]',            # Any array
        r'\{.*?\}',            # Any object
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            try:
                # Test if it's valid JSON
                json_str = match.group(0)
                json.loads(json_str)
                return json_str
            except:
                continue
    
    return None


def merge_menu_items(items_list: List[List[dict]]) -> List[dict]:
    """
    Merge menu items from multiple pages, deduplicating by name.
    
    Args:
        items_list: List of item lists from different pages
        
    Returns:
        Merged and deduplicated item list
    """
    seen_names = set()
    merged = []
    
    for page_items in items_list:
        for item in page_items:
            # Normalize name for comparison
            name_normalized = item.get('name', '').lower().strip()
            if name_normalized and name_normalized not in seen_names:
                seen_names.add(name_normalized)
                merged.append(item)
    
    return merged
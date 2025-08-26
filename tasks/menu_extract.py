#!/usr/bin/env python3
"""
CLI tool for extracting menus from PDF files.

Usage:
    python -m tasks.menu_extract                      # process all PDFs in opt/menu/
    python -m tasks.menu_extract --file opt/menu/x.pdf
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path to import parlaplate
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from dotenv import load_dotenv

from parlaplate.extract import extract_menu_from_pdf_path
from parlaplate.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    VISION_EXTRACTION_USER_PROMPT,
    RESTAURANT_SUMMARY_SYSTEM
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from environment or .env file."""
    load_dotenv()
    
    config = {
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'model_chat': os.getenv('OPENAI_MODEL_CHAT', 'gpt-4'),
        'model_vision': os.getenv('OPENAI_MODEL_VISION', 'gpt-4-vision-preview'),
    }
    
    if not config['openai_api_key']:
        logger.error("OPENAI_API_KEY not found in environment or .env file")
        sys.exit(1)
    
    return config


def find_pdf_files(directory: str = "opt/menu") -> list:
    """Find all PDF files in directory."""
    if not os.path.exists(directory):
        logger.warning(f"Directory {directory} does not exist")
        return []
    
    pdf_files = []
    for filename in os.listdir(directory):
        if filename.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(directory, filename))
    
    return sorted(pdf_files)


def process_pdf_file(pdf_path: str, client: OpenAI, config: dict) -> bool:
    """
    Process a single PDF file.
    
    Args:
        pdf_path: Path to PDF file
        client: OpenAI client
        config: Configuration dict
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Processing: {pdf_path}")
        
        # Extract menu
        menu_json, output_path = extract_menu_from_pdf_path(
            pdf_path=pdf_path,
            client=client,
            model_vision=config['model_vision'],
            model_chat=config['model_chat'],
            extraction_system=EXTRACTION_SYSTEM_PROMPT,
            extraction_user=VISION_EXTRACTION_USER_PROMPT,
            summary_system=RESTAURANT_SUMMARY_SYSTEM
        )
        
        # Print stats
        restaurant_name = menu_json.restaurant.display_name or menu_json.restaurant.name or "Unknown"
        logger.info(f"✅ Extracted menu for: {restaurant_name}")
        logger.info(f"   📄 Total items: {len(menu_json.items)}")
        logger.info(f"   🏷️  Categories: {len(set(item.category for item in menu_json.items if item.category))}")
        logger.info(f"   💰 Price level: {menu_json.restaurant.price_level or 'unknown'}")
        logger.info(f"   📁 Saved to: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error processing {pdf_path}: {e}")
        return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract restaurant menus from PDF files using GPT Vision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tasks.menu_extract                      # Process all PDFs in opt/menu/
  python -m tasks.menu_extract --file menu.pdf     # Process specific file
  python -m tasks.menu_extract --input-dir custom/ # Process PDFs in custom directory
        """
    )
    
    parser.add_argument(
        '--file',
        help='Process a specific PDF file'
    )
    
    parser.add_argument(
        '--input-dir',
        default='opt/menu',
        help='Directory containing PDF files (default: opt/menu)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='opt/menu_content',
        help='Output directory for JSON files (default: opt/menu_content)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = load_config()
    
    # Create OpenAI client
    client = OpenAI(api_key=config['openai_api_key'])
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine files to process
    if args.file:
        if not os.path.exists(args.file):
            logger.error(f"File not found: {args.file}")
            sys.exit(1)
        pdf_files = [args.file]
    else:
        pdf_files = find_pdf_files(args.input_dir)
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {args.input_dir}")
            logger.info(f"Place PDF files in {args.input_dir}/ and run again")
            return
    
    logger.info(f"Found {len(pdf_files)} PDF file(s) to process")
    
    # Process files
    successful = 0
    failed = 0
    
    for pdf_file in pdf_files:
        if process_pdf_file(pdf_file, client, config):
            successful += 1
        else:
            failed += 1
        
        # Add a separator between files
        if len(pdf_files) > 1:
            logger.info("-" * 50)
    
    # Final summary
    logger.info(f"📊 Processing complete:")
    logger.info(f"   ✅ Successful: {successful}")
    logger.info(f"   ❌ Failed: {failed}")
    logger.info(f"   📁 Output directory: {args.output_dir}")
    
    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
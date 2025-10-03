#!/usr/bin/env python
"""
Quarto filter to handle {{< path >}} shortcode for site-relative paths.

This filter resolves paths relative to the site root, making them work 
correctly from pages in any directory level (including subdirectories 
like chapters/).

Usage: {{< path images/logo.png >}} → proper relative path
"""

import panflute as pf
import re
from pathlib import Path


def prepare(doc):
    """Initialize document-level variables."""
    # Get the site URL from metadata if available
    doc.site_url = doc.get_metadata("site-url", "")


def action(elem, doc):
    """
    Handle {{< path filename >}} shortcodes in various element types.
    
    Converts shortcodes to proper relative paths that work from any page location.
    """
    # Handle different element types that might contain shortcodes
    text_content = None
    
    if isinstance(elem, pf.Str):
        text_content = elem.text
    elif isinstance(elem, pf.RawInline):
        text_content = elem.text
    elif isinstance(elem, pf.RawBlock):
        text_content = elem.text
    elif hasattr(elem, 'text'):
        text_content = elem.text
    
    if text_content:
        # Match {{< path filename >}} pattern (flexible whitespace)
        path_pattern = r'\{\{\s*<\s*path\s+([^>]+?)\s*>\s*\}\}'
        
        def replace_shortcode(match):
            # Clean up the path (remove quotes if present)
            clean_path = match.group(1).strip().strip('"').strip("'")
            
            # For GitHub Pages served from root, use absolute path
            # This works consistently across all page locations
            resolved_path = f"/{clean_path}"
            return resolved_path
        
        new_text = re.sub(path_pattern, replace_shortcode, text_content)
        
        # Update the element if changes were made
        if new_text != text_content:
            if hasattr(elem, 'text'):
                elem.text = new_text
            return elem
    
    return None


def main(doc=None):
    """Main filter function."""
    return pf.run_filter(action, prepare=prepare, doc=doc)


if __name__ == "__main__":
    main()
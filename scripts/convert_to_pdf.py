#!/usr/bin/env python3
"""
Convert Markdown documentation to PDF with embedded images
"""

import os
import re
import base64
import markdown2
from weasyprint import HTML, CSS
from pathlib import Path

def embed_images_as_base64(html_content, base_path):
    """Replace image src paths with base64 encoded data"""
    
    def replace_img(match):
        img_tag = match.group(0)
        src_match = re.search(r'src=["\']([^"\']+)["\']', img_tag)
        if src_match:
            img_path = src_match.group(1)
            # Handle relative paths
            if not img_path.startswith(('http://', 'https://', 'data:')):
                full_path = os.path.join(base_path, img_path)
                if os.path.exists(full_path):
                    # Determine mime type
                    ext = os.path.splitext(img_path)[1].lower()
                    mime_types = {
                        '.png': 'image/png',
                        '.jpg': 'image/jpeg',
                        '.jpeg': 'image/jpeg',
                        '.gif': 'image/gif',
                        '.svg': 'image/svg+xml'
                    }
                    mime_type = mime_types.get(ext, 'image/png')
                    
                    # Read and encode image
                    with open(full_path, 'rb') as f:
                        img_data = base64.b64encode(f.read()).decode('utf-8')
                    
                    # Replace src with data URI
                    new_src = f'data:{mime_type};base64,{img_data}'
                    img_tag = img_tag.replace(src_match.group(0), f'src="{new_src}"')
                else:
                    print(f"Warning: Image not found: {full_path}")
        return img_tag
    
    return re.sub(r'<img[^>]+>', replace_img, html_content)

def convert_md_to_pdf(md_path, output_pdf_path):
    """Convert Markdown file to PDF"""
    
    base_path = os.path.dirname(os.path.abspath(md_path))
    
    # Read markdown content
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown2.markdown(
        md_content,
        extras=[
            'tables',
            'fenced-code-blocks',
            'code-friendly',
            'header-ids',
            'toc',
            'cuddled-lists',
            'metadata',
            'strike',
            'task_list'
        ]
    )
    
    # Embed images as base64
    html_content = embed_images_as_base64(html_content, base_path)
    
    # Create full HTML document with styling
    full_html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Cross-Modal Integration Study</title>
    <style>
        @page {{
            size: A4;
            margin: 2cm 1.5cm;
            @top-center {{
                content: "Apparent Inefficiency, Hidden Optimality";
                font-size: 9pt;
                color: #666;
            }}
            @bottom-center {{
                content: counter(page);
                font-size: 9pt;
            }}
        }}
        
        body {{
            font-family: "Times New Roman", Times, serif;
            font-size: 11pt;
            line-height: 1.5;
            color: #333;
            max-width: 100%;
            text-align: justify;
        }}
        
        h1 {{
            font-size: 18pt;
            font-weight: bold;
            color: #1a1a1a;
            margin-top: 24pt;
            margin-bottom: 12pt;
            text-align: center;
            page-break-after: avoid;
        }}
        
        h2 {{
            font-size: 14pt;
            font-weight: bold;
            color: #2a2a2a;
            margin-top: 18pt;
            margin-bottom: 10pt;
            border-bottom: 1px solid #ccc;
            padding-bottom: 4pt;
            page-break-after: avoid;
        }}
        
        h3 {{
            font-size: 12pt;
            font-weight: bold;
            color: #3a3a3a;
            margin-top: 14pt;
            margin-bottom: 8pt;
            page-break-after: avoid;
        }}
        
        h4 {{
            font-size: 11pt;
            font-weight: bold;
            color: #4a4a4a;
            margin-top: 12pt;
            margin-bottom: 6pt;
            page-break-after: avoid;
        }}
        
        p {{
            margin-bottom: 10pt;
            text-indent: 0;
        }}
        
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 16pt auto;
            page-break-inside: avoid;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 12pt 0;
            font-size: 9pt;
            page-break-inside: avoid;
        }}
        
        th, td {{
            border: 1px solid #ddd;
            padding: 6pt 8pt;
            text-align: left;
        }}
        
        th {{
            background-color: #f5f5f5;
            font-weight: bold;
        }}
        
        tr:nth-child(even) {{
            background-color: #fafafa;
        }}
        
        code {{
            font-family: "Courier New", Courier, monospace;
            font-size: 9pt;
            background-color: #f4f4f4;
            padding: 1pt 3pt;
            border-radius: 2pt;
        }}
        
        pre {{
            background-color: #f4f4f4;
            padding: 10pt;
            border-radius: 4pt;
            overflow-x: auto;
            font-size: 9pt;
            page-break-inside: avoid;
        }}
        
        blockquote {{
            border-left: 3pt solid #ccc;
            margin: 12pt 0;
            padding: 8pt 16pt;
            background-color: #f9f9f9;
            font-style: italic;
            page-break-inside: avoid;
        }}
        
        ul, ol {{
            margin-bottom: 10pt;
            padding-left: 20pt;
        }}
        
        li {{
            margin-bottom: 4pt;
        }}
        
        hr {{
            border: none;
            border-top: 1px solid #ccc;
            margin: 20pt 0;
        }}
        
        strong {{
            font-weight: bold;
        }}
        
        em {{
            font-style: italic;
        }}
        
        a {{
            color: #0066cc;
            text-decoration: none;
        }}
        
        /* Table of Contents styling */
        .toc {{
            background-color: #f9f9f9;
            padding: 16pt;
            border-radius: 4pt;
            margin: 16pt 0;
        }}
        
        /* Keywords styling */
        .keywords {{
            font-style: italic;
            color: #666;
        }}
        
        /* Figure captions */
        .figure-caption {{
            text-align: center;
            font-size: 10pt;
            color: #666;
            margin-top: 8pt;
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>'''
    
    # Save HTML for debugging
    html_output = output_pdf_path.replace('.pdf', '.html')
    with open(html_output, 'w', encoding='utf-8') as f:
        f.write(full_html)
    print(f"HTML saved to: {html_output}")
    
    # Convert to PDF
    print("Converting to PDF...")
    HTML(string=full_html, base_url=base_path).write_pdf(output_pdf_path)
    print(f"PDF saved to: {output_pdf_path}")
    
    return output_pdf_path

if __name__ == "__main__":
    # Paths
    project_dir = "/root/autodl-fs/CCN_Competition/project_1"
    md_file = os.path.join(project_dir, "PROJECT_COMPREHENSIVE_DOCUMENTATION.md")
    pdf_file = os.path.join(project_dir, "PROJECT_COMPREHENSIVE_DOCUMENTATION.pdf")
    
    # Convert
    convert_md_to_pdf(md_file, pdf_file)
    print("\nConversion complete!")


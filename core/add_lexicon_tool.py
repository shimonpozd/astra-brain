import re

# Read the file
with open('brain_service/core/startup.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Define the lexicon tool schema and registration
lexicon_addition = """
    # Lexicon tool for word definitions
    sefaria_get_lexicon_schema = {
        "type": "function",
        "function": {
            "name": "sefaria_get_lexicon",
            "description": "Get word definition and linguistic explanation from Sefaria lexicon. Use when user asks about meaning of Hebrew/Aramaic words.",
            "parameters": {
                "type": "object",
                "properties": {
                    "word": {
                        "type": "string",
                        "description": "Hebrew or Aramaic word to look up, e.g., 'שבת' or 'תלמוד'"
                    }
                },
                "required": ["word"]
            }
        }
    }
    app.state.tool_registry.register(
        name="sefaria_get_lexicon",
        handler=app.state.lexicon_service.get_word_definition,
        schema=sefaria_get_lexicon_schema
    )
"""

# Find the position after the sefaria_get_related_links registration
pattern = r'(app\.state\.tool_registry\.register\(\s*name="sefaria_get_related_links",.*?schema=sefaria_get_related_links_schema\s*\))'
match = re.search(pattern, content, re.DOTALL)

if match:
    # Insert the lexicon addition after the related_links registration
    insert_pos = match.end()
    new_content = content[:insert_pos] + lexicon_addition + content[insert_pos:]
    
    # Write the updated content
    with open('brain_service/core/startup.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print('Successfully added sefaria_get_lexicon tool to startup.py')
else:
    print('Could not find the insertion point in startup.py')

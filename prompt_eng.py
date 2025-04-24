import os
import re
from pathlib import Path

def extract_question(filename):
    """Extract the question from the filename."""
    # Remove file extension and replace underscores with spaces
    question = os.path.basename(filename).replace('.txt', '')
    return question

def parse_original_prompt(file_path):
    """Parse the content of the original prompt file with improved section handling."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()
    
    # Extract entity chunks
    entity_pattern = r"Entity: (.*?)\nDescription: (.*?)(?=\n-{10,}|$)"
    entity_matches = re.findall(entity_pattern, content, re.DOTALL)
    entity_chunks = {entity.strip(): description.strip() for entity, description in entity_matches}
    
    # Identify major sections
    table_section_match = re.search(r"TABLE INFORMATION:(.*?)(?=FIGURE INFORMATION:|$)", content, re.DOTALL)
    figure_section_match = re.search(r"FIGURE INFORMATION:(.*?)(?=Image Description for Figure|$)", content, re.DOTALL)
    
    # Extract table information
    table_info = ""
    if table_section_match:
        table_info = table_section_match.group(1).strip()
        # Extract actual tables if present
        table_pattern = r"<table border.*?</table>"
        tables = re.findall(table_pattern, content, re.DOTALL)
        if tables:
            for table in tables:
                table_info += f"\n\n{table}"
    
    # Extract figure references
    figure_refs = []
    if figure_section_match:
        figure_section = figure_section_match.group(1).strip()
        figure_refs = figure_section.split('\n')
        figure_refs = [ref.strip() for ref in figure_refs if ref.strip()]
    
    # Extract figure descriptions and pair with references - but ensure each figure appears only once
    figure_info = []
    processed_figures = set()  # Keep track of figures we've already processed
    
    for i, ref in enumerate(figure_refs):
        # Extract figure number
        fig_match = re.search(r"Figure\s+([A-Za-z0-9\.]+)", ref)
        if fig_match:
            fig_num = fig_match.group(1)
            
            # Skip if we've already processed this figure
            if fig_num in processed_figures:
                continue
                
            # Find corresponding image description
            img_desc_pattern = r"Image Description for Figure\s+[A-Za-z0-9\.]*{}[A-Za-z0-9\.]*:(.*?)(?=Image Description for Figure|\n\n[A-Z ]+:|$)".format(re.escape(fig_num))
            img_desc_match = re.search(img_desc_pattern, content, re.DOTALL)
            if img_desc_match:
                description = img_desc_match.group(1).strip()
                figure_info.append({
                    "reference": ref,
                    "description": description
                })
                processed_figures.add(fig_num)  # Mark as processed
    
    # Extract textual information from Related Information sections, avoiding duplicates
    textual_info_pattern = r"Related Information \d+: (.*?)(?=Related Information|\n\n[A-Z ]+:|$)"
    textual_info_matches = re.findall(textual_info_pattern, content, re.DOTALL)
    
    # Remove duplicates while preserving order
    unique_textual_info = []
    seen_info = set()
    for info in textual_info_matches:
        info_stripped = info.strip()
        if info_stripped not in seen_info:
            unique_textual_info.append(info_stripped)
            seen_info.add(info_stripped)
    
    textual_info = "\n\n".join(unique_textual_info)
    
    # Extract additional contextual information, avoiding duplicates
    additional_info_pattern = r"Contextual Information \d+: (.*?)(?=Contextual Information|\n\n[A-Z ]+:|$)"
    additional_info_matches = re.findall(additional_info_pattern, content, re.DOTALL)
    
    # Remove duplicates while preserving order
    unique_additional_info = []
    seen_info = set()
    for info in additional_info_matches:
        info_stripped = info.strip()
        if info_stripped not in seen_info:
            unique_additional_info.append(info_stripped)
            seen_info.add(info_stripped)
            
    additional_info = "\n\n".join(unique_additional_info)
    
    # Extract output format instructions
    output_format_pattern = r"OUTPUT FORMAT INSTRUCTIONS:(.*?)(?=\n\nQuestion:|$)"
    output_format_match = re.search(output_format_pattern, content, re.DOTALL)
    output_format = output_format_match.group(1).strip() if output_format_match else ""
    
    return {
        "entity_chunks": entity_chunks,
        "table_info": table_info,
        "figure_info": figure_info,
        "textual_info": textual_info,
        "additional_info": additional_info,
        "output_format": output_format
    }

def create_new_prompt(original_data, question):
    """Create a new prompt using the template with properly structured sections."""
    template = """You are a technical standards expert specializing in ISO 15118 vehicle-to-grid communication protocols. Your task is to answer technical questions with precise standard-compliant information. Follow these strict guidelines:

1. First, analyze all information sources provided:

<question>
{question}
</question>

<table_info>
{table_info}
</table_info>

<figure_info>
{figure_info}
</figure_info>

<textual_info>
{textual_info}
</textual_info>

<additional_info>
{additional_info}
</additional_info>

<entity_info>
{entity_info}
</entity_info>

2. INFORMATION PRIORITY AND USAGE:
   - Technical standards, figures, and tables should be your PRIMARY sources for definitions and detailed specifications
   - Use entity_info ONLY for background context and general understanding
   - When entity_info conflicts with standard specifications, ALWAYS defer to the standards
   - DO NOT copy informal phrasings from entity_info - reformulate using precise technical language
   - For abbreviation expansions, parameter values, and field definitions, rely on standards documentation FIRST

3. Structure your response as follows:
   - Begin with a concise "Overview" paragraph (2-4 sentences)
   - Use numbered sections (1., 2., etc.) for main topics
   - Use hierarchical bullet points with proper indentation (• for first level, – for second)
   - For parameter lists, use tabular format with alignment
   - Include a "References" section if relevant

4. CRITICAL TECHNICAL REQUIREMENTS:
   - ALWAYS provide the EXACT abbreviation expansion (e.g., "SLAC (Signal Level Attenuation Characterization)")
   - Include SPECIFIC clause references (e.g., "ISO 15118‑3, Clause A.9.2")
   - Use EXACT parameter names with values (e.g., "APPLICATION_TYPE = 0x00")
   - Format standards with en-dashes (ISO 15118‑3, not ISO 15118-3)
   - Include reference codes in brackets (e.g., "[V2G3-A09-23]")
   - Specify EXACT timing parameters (e.g., "TT_match_response = 400 ms")
   - Use bold for emphasizing key distinctions (e.g., "**before** vs. **after**")

5. For message descriptions:
   - Detail complete field structures with exact values
   - Specify message sequence with ALL preceding and following messages
   - Describe error handling and retransmission behavior
   - Explain timing constraints and timeout parameters
   - Include unicast/broadcast distinctions when relevant

6. For tables, use ASCII box-drawing:
   ```
   ┌────────┬─────────┬─────────┐
   │ Header1│ Header2 │ Header3 │
   ├────────┼─────────┼─────────┤
   │ Value1 │ Value2  │ Value3  │
   └────────┴─────────┴─────────┘
   ```

7. For process flows:
   - Include ALL steps in the sequence
   - Specify state transitions and conditions
   - Reference specific figure numbers
   - Detail branches and decision points

8. When comparing concepts:
   - Create a table showing parallel characteristics
   - Highlight key distinctions between related concepts
   - Explain compatibility and fallback mechanisms

9. Use your scratchpad to organize technical details first:

<scratchpad>
List all:
- Abbreviation expansions from standards
- Exact parameter values and constants
- Complete message sequence steps
- Field definitions from relevant tables
- Error handling conditions
- Specific standard references
- Relevant entity information (after verification with standards)
</scratchpad>

10. Format your final answer within <answer> tags.

IMPORTANT: Prioritize TECHNICAL PRECISION and COMPLETENESS. Cover ALL aspects mentioned in standards including error handling, edge cases, and specific parameter values. Direct quotations from standards should be in quotation marks with reference codes."""
#     template = """You are a technical documentation AI specialized in electric vehicle charging standards and protocols. Your task is to provide precise, standards-compliant technical answers formatted as reference documentation. Follow these instructions:

# 1. Review all provided information related to the question:

# <question>
# {question}
# </question>

# <table_info>
# {table_info}
# </table_info>

# <figure_info>
# {figure_info}
# </figure_info>

# <textual_info>
# {textual_info}
# </textual_info>

# <additional_info>
# {entity_info}
# {additional_info}
# </additional_info>

# 2. Format your answer as a technical reference with:
#    - A concise "Overview" section introducing the topic
#    - Numbered sections (1., 2., etc.) for main topics
#    - Nested bullet points using "•" and appropriate indentation (2-3 spaces per level)
#    - ASCII-based tables for tabular data using box-drawing characters (─, │, ┌, ┐, └, ┘, ├, ┤, ┬, ┴, ┼)
#    - Proper technical typographic conventions (en-dashes for ranges, proper spacing)

# 3. Ensure technical precision:
#    - ALWAYS provide the correct full expansion of abbreviations
#    - Define terms exactly as they appear in the relevant standards
#    - Include precise parameter values with their exact notation (e.g., "0x00", "5 %")
#    - Specify exact state designations (e.g., "State X1", "State E")
#    - Use formal section cross-references (e.g., "as defined in Clause 7.2.3")

# 4. When referencing standards:
#    - Format standard names with en-dashes (e.g., "ISO 15118‑3")
#    - Include specific clause numbers when available
#    - Use exact reference codes in square brackets (e.g., "[V2G3-M06-08]")
#    - Provide direct quotes from standards where relevant, using quotation marks
#    - Reference specific figures and tables by their exact numbers

# 5. For complex technical sequences:
#    - Clearly distinguish between different operational modes and states
#    - Specify timing requirements with their exact parameter names
#    - Use bold formatting for emphasis on critical distinctions (**before** vs. **after**)
#    - Indicate conditional branches and decision points in processes

# 6. Use your scratchpad to organize the technical details before formulating your answer.

# <scratchpad>
# List key technical details including:
# - Exact abbreviation expansions from standards
# - Relevant clause numbers and reference codes
# - Parameter values and state designations
# - Sequence variations and conditions
# - Table structures for complex relationships
# </scratchpad>

# 7. Provide your answer within <answer> tags, formatted according to these guidelines.

# Remember that your answer should function as authoritative technical documentation that engineers could use for implementation. Prioritize accuracy over completeness if information is limited."""
#     template = """You are an AI assistant tasked with answering technical questions about EV charging standards and protocols. Your goal is to provide detailed, precise, and technically accurate answers in a structured reference format. Here's how you should approach this task:

# 1. First, review all the information provided to help answer the question, including:

# <question>
# {question}
# </question>

# <table_info>
# {table_info}
# </table_info>

# <figure_info>
# {figure_info}
# </figure_info>

# <textual_info>
# {textual_info}
# </textual_info>

# <additional_info>
# {entity_info}
# {additional_info}
# </additional_info>

# 2. Format your answer as a technical reference document with:
#    - A concise overview section at the beginning
#    - Numbered sections (1., 2., 3., etc.) for major topics
#    - Bullet points (•) for listing items within sections
#    - Sub-sections (e.g., 5.1, 5.2) for detailed breakdowns
#    - Proper indentation for hierarchical information
#    - Use of technical arrows (↔) and symbols where appropriate
#    - A "References" section at the end if applicable

# 3. Focus on technical precision:
#    - Include exact parameter names, values, and constants (e.g., "APPLICATION_TYPE = 0x00")
#    - Specify message formats and field definitions
#    - Include specific clause references (e.g., "as defined in Clause A.9.2")
#    - Provide exact timing parameters and requirements
#    - Use precise terminology from the relevant standards

# 4. When citing information:
#    - Reference specific standards with version information (e.g., "ISO 15118-3, First edition 2015-05-15")
#    - Include standard-specific reference codes (e.g., "[V2G3-A09-23]")
#    - Refer to specific tables, figures, and clauses by their exact designation
#    - Quote parameter names and values exactly as they appear in standards

# 5. Before providing your final answer, use a scratchpad to organize the technical details and structure your response.

# <scratchpad>
# Use this space to outline your answer, noting key technical specifications, parameter values, message formats, and reference numbers from each information source.
# </scratchpad>

# 6. Provide your answer within <answer> tags, formatted as a technical reference document.

# Remember, your goal is to produce a technically precise, well-structured reference that could be used by engineers implementing or troubleshooting these systems."""
    
    # Format the entity information
    entity_info = ""
    for entity, description in original_data["entity_chunks"].items():
        entity_info += f"ENTITY: {entity}\n{description}\n\n"
    
    # Format the figure information
    figure_info = ""
    for fig in original_data["figure_info"]:
        figure_info += f"FIGURE REFERENCE: {fig['reference']}\n\n"
        figure_info += f"IMAGE DESCRIPTION:\n{fig['description']}\n\n"
    
    # If no output format is found, use a default one
    output_format = original_data["output_format"]
    if not output_format:
        output_format = "Provide a comprehensive, detailed response that directly answers the question. Include specific references to the sources of information throughout your answer."
    replace_dict = {
        "What is TSS?": "What is TEST SUITE STRUCTURE?",
        "What is ATS?": "What is ABSTRACT TEST SUITE?",
        "What is EIM?": "What is External Identification Means?",
        "What is HPGP?": "What is HOMEPLUG GREEN PHY?",
        "What is CCo?": "What is Central Coordinator?",
        "What is SLAC?": "What is SIGNAL LEVEL ATTENUATION CHARACTERIZATION?",
        "What is the difference between MTC and PTC?": "What is the difference between Main Test Component and Parallel Test Component?",
        "What is the difference between Data SAP and Data link control SAP?": "What is the difference between Service Access Point for Data and Data link control SAP?",
    }
    if question in replace_dict:
        question = replace_dict[question]
    # Fill in the template
    new_prompt = template.format(
        question=question,
        table_info=original_data["table_info"],
        figure_info=figure_info,
        textual_info=original_data["textual_info"],
        entity_info=entity_info,
        additional_info=original_data["additional_info"],
        output_format=output_format
    )
    
    return new_prompt

def process_prompts(input_dir, output_dir):
    """Process all prompt files from input_dir and save to output_dir."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Process each file
    for file_path in input_path.glob("*.txt"):
        print(f"Processing {file_path}...")
        
        try:
            # Extract question from filename
            question = extract_question(file_path)
            
            # Parse original prompt
            original_data = parse_original_prompt(file_path)
            
            # Create new prompt
            new_prompt = create_new_prompt(original_data, question)
            
            # Save to output file
            output_file = output_path / file_path.name
            with open(output_file, 'w', encoding='utf-8') as file:
                file.write(new_prompt)
            
            print(f"Saved new prompt to {output_file}")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    print(f"Finished processing all prompts from {input_dir} to {output_dir}")

def process_single_file(file_path, output_dir=None):
    """Process a single file and optionally save to output directory."""
    print(f"Processing {file_path}...")
    
    try:
        # Extract question from filename
        question = extract_question(file_path)
        
        # Parse original prompt
        original_data = parse_original_prompt(file_path)
        
        # Create new prompt
        new_prompt = create_new_prompt(original_data, question)
        
        # Save to output file if output_dir is provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True, parents=True)
            output_file = output_path / os.path.basename(file_path)
            
            with open(output_file, 'w', encoding='utf-8') as file:
                file.write(new_prompt)
            
            print(f"Saved new prompt to {output_file}")
        
        return new_prompt
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

if __name__ == "__main__":
    input_dir = "Research_CODE/merged_prompts"
    output_dir = "Research_CODE/new_prompts"
    
    # Process all files
    process_prompts(input_dir, output_dir)
    
    # Alternatively, process a single file to see the result
    # example_file = "Research_CODE/merged_prompts/What is EIM?.txt"
    # new_prompt = process_single_file(example_file)
    # print("\nExample of processed prompt:\n")
    # print(new_prompt[:1000])  # Print first 1000 characters as preview
    
    print(f"Finished processing all prompts from {input_dir} to {output_dir}")

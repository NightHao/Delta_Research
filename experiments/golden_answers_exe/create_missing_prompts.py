#!/usr/bin/env python3
"""
Create prompts for missing questions like ATS
"""
import os
import json
import sys

def load_json_file(file_path):
    """Load data from a JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded data from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return None

def save_prompt(question, content, output_dir):
    """Save a prompt to a file"""
    # Create a clean filename
    filename = question.lower().replace(' ', '_').replace('?', '').replace(':', '').replace('/', '_')
    filename = ''.join(c for c in filename if c.isalnum() or c == '_')
    filename = filename[:50] + '.txt'  # Limit length
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the prompt
    output_file = os.path.join(output_dir, filename)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Created prompt for {question}: {output_file}")

def create_prompt_content(question, information=None):
    """Create content for a prompt"""
    if not information:
        return f"Question {question}: {question}\n\nNo information available for this question."
    
    # Build the prompt
    prompt = f"Question {question}: {question}\n\n"
    prompt += "Below is all the available information related to this question.\n"
    prompt += "Use ONLY this information to construct a comprehensive answer.\n"
    prompt += "Do not add any new information that is not present below.\n\n"
    
    # Add the information
    prompt += "TEXTUAL INFORMATION:\n"
    for i, info in enumerate(information, 1):
        prompt += f"Text {i}: {info}\n"
    prompt += "\n"
    
    # Add output format instructions
    prompt += "OUTPUT FORMAT INSTRUCTIONS:\n"
    prompt += "Your answer should follow these guidelines:\n"
    prompt += "1. Start with an 'Overview' section that provides a concise explanation of what this concept/topic is.\n"
    prompt += "2. For the related content, organize it into appropriate sections with descriptive headings based on the information available.\n"
    prompt += "   - Analyze the content and group related information logically.\n"
    prompt += "   - Choose section titles that best represent the information, such as 'Related Topics', 'Related Primitives', 'Configuration Requirements', etc.\n"
    prompt += "   - Let the content guide your choice of section titles rather than forcing a predefined structure.\n"
    prompt += "3. The structure should match the format used in technical documentation, with clear hierarchy and organization.\n"
    prompt += "4. Include any relevant figures, tables, or processes mentioned in the information.\n"
    prompt += "5. Be as detailed and comprehensive as possible while ONLY using the provided information.\n"
    prompt += "6. In writing your answer, strictly follow the content provided in the information, and do not add any new information or make assumptions.\n"
    
    return prompt

def main():
    # Missing questions and their information
    missing_questions = {
        "What is ATS?": [
            "Abstract Test Suite",
            "test suite composed of abstract test cases",
            "The capability tests within the ATS check that the implementation has the capability of supporting the functionality described in ISO 15118-3.",
            "The following conditions apply in terms of test case coverage:",
            "— Requirements that are out of scope according to ISO 15118-5 are excluded from testing.",
            "— Requirements that are not explicitly tested through dedicated test cases have been covered implicitly.",
            "The resulting test suite coverage in this document is indicated as follows:",
            "X Indicates requirements that are covered in the ATS through dedicated test cases.",
            "I Indicates requirements that are indirectly covered through other test cases.",
            "P Indicates requirements that are only partially covered, i.e. some aspects are tested but not all.",
            "N Requirements that are not testable for the profile.",
            "‐ Requirements that are not applicable for the profile.",
            "O Requirements that are out of scope for the profile.",
            "Protocol Implementation Conformance Statement",
            "To evaluate the conformance of a particular implementation to ISO 15118-3, an implementation has to specify its capabilities and options with respect to ISO 15118-3.",
            "All PICS defined in the ATS are summarized in Tables 6 to 8.",
            "Protocol implementation extra information for testing",
            "In order to ensure testability of ISO 15118‐3 requirements, additional information about the implementation is needed that is not necessarily required for conformance but for test execution."
        ],
        "What is TSS?": [
            "test suite structure",
            "A test suite is a complete set of test cases, possibly combined into nested test groups that are needed to perform conformance testing.",
            "In each test case, the SUT is stimulated with specified protocol events in a specified context and the SUT's reaction is analyzed whether it complies with the requirements.",
            "This subclause defines test profiles for conformance testing with ISO 15118-3."
        ],
        "What is the difference between MTC and PTC?": [
            "MTC Main Test Component",
            "parallel test component",
            "MTC single test component in a test component configuration responsible for the test verdict",
            "test component created by the main test component",
            "The MTC always contains a TTCN‐3 test configurations such as TSI, HAL components, the protocol stack (IP, TCP, TLS, XML, and all V2G messages) which is required for all test cases.",
            "The parallel test component (PTC) for IEC 61851‐1 may contain all functionality to act as the IEC 61851‐1 counterpart.",
            "These components and ports compose relevant test configurations according to the PICS and PIXIT parameters.",
            "[V2G5-010] If the SUT is an SECC, the MTC shall use the communication and control port to map an EVCC and its behavior.",
            "The HAL_61851_Listener combines all necessary listening interfaces to observe the low-level communication.",
            "As shown in Figure 3 the port mappings are defined by the SUT as follows:",
            "— The port pt_SLAC_Port of the TSI is always mapped to the PLC simulation.",
            "— The port pt_HAL_61851_Port of the TSI is always mapped to either the EV- or EVSE-simulation according to the test configuration.",
            "— The port pt_HAL_61851_Listener_Port of the TSI is always mapped to HAL_61851_Listener.",
            "— The internal port pt_HAL_61851_Internal_Port is always mapped internally to the HAL_61851_Listeneer.",
            "TSI TTCN‐3 System Interface",
            "test component that provides a mapping of the ports to the SUT",
            "In addition to the MTC and PTC and their corresponding ports, the HAL components also have to define ports to real interfaces to the SUT."
        ]
    }
    
    # Check for debug flag
    debug_mode = "--debug" in sys.argv
    if debug_mode:
        print("Running in debug mode")
    
    # Output directory
    output_dir = '/home/yeskey525/Research_CODE/experiments/golden_answers/prompts'
    
    # Create prompts for missing questions
    for question, info in missing_questions.items():
        prompt_content = create_prompt_content(question, info)
        save_prompt(question, prompt_content, output_dir)
    
    print("All missing prompts created successfully!")

if __name__ == "__main__":
    main() 
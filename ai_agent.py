# ai_agent.py

import google.generativeai as genai
import json
import os
import re
import logging
import traceback
from typing import Dict, List, Any
from google.api_core import exceptions
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.generativeai import types, protos # Assuming protos might be needed for exceptions if not in types

# Configure logging for the AI agent to help with debugging API calls
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class SelfImprovingAgent:
    def __init__(self, api_key=None):
        """Initialize the agent with Gemini API, loading key from environment variable."""
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")

        # Initialize model and chat as None initially
        self.model = None
        self.chat = None
        self.api_status = "INITIALIZING"

        if not self.api_key:
            logging.error("GOOGLE_API_KEY environment variable not set or API key not provided.")
            # Use print for the simple case, as this __init__ might not use logging.basicConfig
            print("Please set GOOGLE_API_KEY environment variable before running the application for advanced features.")
            self.api_status = "API_KEY_MISSING"
        else:
            try:
                genai.configure(api_key=self.api_key)
                
                # --- MODIFIED: Added generation_config to increase max_output_tokens ---
                self.model = genai.GenerativeModel(
                    'gemini-1.5-flash',
                    generation_config=genai.GenerationConfig(max_output_tokens=4000) # Increased token limit
                )
                self.chat = self.model.start_chat(history=[])
                logging.info("Gemini model initialized successfully.")
                self.api_status = "READY"
            except Exception as e:
                logging.error(f"Failed to initialize Gemini model: {e}", exc_info=True) # exc_info=True for full traceback
                self.model = None
                self.chat = None
                self.api_status = "API_INIT_FAILED"
                logging.warning("AI Agent feedback and content generation will be limited or unavailable.")
                print(f"Failed to initialize Gemini model: {e}") # Keep print for immediate feedback if logging isn't configured
                print("Feedback will be limited to rule-based or unavailable.")

        self.memory = {
            'successful_strategies': [],
            'failed_attempts': [],
            'learned_patterns': [],
            'performance_metrics': [],
        }
        self.capabilities = {
            'feedback_quality': 0.8, # Placeholder, actual quality from API
            'error_detection': 0.7,
            'hint_relevance': 0.7,
        }
        self.iteration_count = 0
        
    # Add the update_model and update_max_tokens methods
    def update_model(self, new_model_name: str):
        """Updates the AI model used for generation."""
        if self.model_name != new_model_name:
            self.model_name = new_model_name
            if self.api_status == "READY": # Only update if agent is functional
                try:
                    self.model = genai.GenerativeModel(
                        model_name=self.model_name,
                        generation_config=genai.GenerationConfig(max_output_tokens=self.max_output_tokens)
                    )
                    logging.info(f"AI Agent model updated to: {new_model_name}")
                except Exception as e:
                    logging.error(f"Error updating AI model to {new_model_name}: {e}")
                    # Optionally set status to error if update fails
                    self.api_status = f"MODEL_UPDATE_ERROR: {e}"

    def update_max_tokens(self, new_max_tokens: int):
        """Updates the maximum output tokens for AI generation."""
        if self.max_output_tokens != new_max_tokens:
            self.max_output_tokens = new_max_tokens
            if self.api_status == "READY": # Only update if agent is functional
                try:
                    # Reinitialize the model with the new max_output_tokens
                    self.model = genai.GenerativeModel(
                        model_name=self.model_name,
                        generation_config=genai.GenerationConfig(max_output_tokens=self.max_output_tokens)
                    )
                    logging.info(f"AI Agent max output tokens updated to: {new_max_tokens}")
                except Exception as e:
                    logging.error(f"Error updating AI max output tokens to {new_max_tokens}: {e}")
                    # Optionally set status to error if update fails
                    self.api_status = f"TOKENS_UPDATE_ERROR: {e}"

    def _generate_content_with_error_handling(self, prompt_parts: List[str], **kwargs) -> str:
        """Helper to call Gemini API with robust error handling."""

        if self.api_status == "API_KEY_MISSING":
            return "Agent Feedback: Gemini API key not found. Please set the GOOGLE_API_KEY environment variable for advanced features."
        if self.api_status == "API_INIT_FAILED" or not hasattr(self, 'model') or not self.model:
            return "Agent Feedback: Gemini API could not be initialized. Check your internet connection or API key validity."

        default_generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 800,
        }

        default_safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        try:
            final_generation_config = {
                **default_generation_config,
                **kwargs.get('generation_config', {})
            }
            final_safety_settings = kwargs.get('safety_settings', default_safety_settings)

            response = self.model.generate_content(
                prompt_parts,
                generation_config=final_generation_config,
                safety_settings=final_safety_settings
            )

            if response and response.text:
                return response.text
            elif response and response.prompt_feedback and response.prompt_feedback.block_reason:
                block_reason = response.prompt_feedback.block_reason
                print(f"Prompt was blocked by safety settings: {block_reason}")
                return f"Your request was blocked by safety settings ({block_reason}). Please try rephrasing."
            else:
                return "Could not generate a response (empty or unexpected API response)."

        except exceptions.ResourceExhausted as e:
            print(f"API Rate Limit Exceeded: {e}")
            return f"API Rate Limit Exceeded. Please wait a moment and try again. ({e})"
        except exceptions.FailedPrecondition as e:
            if "API key not valid" in str(e):
                print(f"Invalid API Key: {e}")
                return f"Invalid API Key. Please double-check your GOOGLE_API_KEY environment variable. ({e})"
            else:
                print(f"Gemini API Precondition Failed: {e}")
                return f"A Gemini API error occurred. Check input or model status. ({e})"
        except exceptions.GoogleAPIError as e:
            print(f"Gemini API Error: {e}")
            return f"A Gemini API error occurred. Please try again later. ({e})"
        except Exception as e:
            print(f"Unexpected error calling Gemini API: {e}")
            return f"An unexpected error occurred while generating feedback. ({e})"


    def provide_feedback_on_code(self, user_code: str, problem_description: str,
                                 expected_output: Any = None, previous_errors: str = "",
                                 num_attempts: int = 1) -> str:
        """Provides intelligent feedback using the Gemini API."""
        prompt_parts = [
            f"You are an expert Python programming tutor helping a beginner. The user is attempting to solve the following problem:\n",
            f"Problem Description: {problem_description}\n",
            f"Expected Output (if applicable): {expected_output}\n",
            f"Here is the user's current code:\n```python\n{user_code}\n```\n",
        ]

        if previous_errors:
            prompt_parts.append(f"The user's code produced the following errors or output when executed:\n```\n{previous_errors}\n```\n")

        prompt_parts.append(f"Given this, provide constructive and encouraging feedback. Point out any errors, suggest improvements, and offer hints without giving away the direct answer. Focus on Python syntax, common pitfalls for beginners, and logical flow. If the code is correct, congratulate them.\n\nFeedback:")

        feedback = self._generate_content_with_error_handling(prompt_parts)
        self.learn_from_experience(False, user_code, "feedback_request", feedback)
        return f"Agent Feedback:\n{feedback}"

    # General AI Question (NO CHANGE)
    def ask_general_question(self, user_query: str) -> str:
        """
        Allows the user to ask the AI general questions about Python or the current task.
        """
        prompt = (
            "You are an expert Python programming tutor. The user has a question "
            "about Python, programming concepts, or their current task. "
            "Provide a helpful and clear answer. Keep it concise but comprehensive. "
            f"User's question: {user_query}\n\nAnswer:"
        )
        response_text = self._generate_content_with_error_handling([prompt])
        return f"AI Response:\n{response_text}"


    def analyze_task(self, task: str) -> Dict[str, Any]:
        """A placeholder for task analysis (would be rule-based or human-curated)."""
        return {
            "complexity": 5,
            "required_skills": ["Python basics", "problem solving"],
            "potential_challenges": ["syntax errors", "logical errors"],
            "recommended_approach": "Break down the problem, write small pieces of code, test often.",
            "success_criteria": "Code produces correct output."
        }

    def learn_from_experience(self, success: bool, user_code: str, problem_id: str, feedback_given: str):
        """
        Conceptual learning: logs successes/failures. For a real product, this would
        involve more sophisticated data storage and analysis.
        """
        if success:
            self.memory['successful_strategies'].append({"problem_id": problem_id, "code_length": len(user_code)})
        else:
            self.memory['failed_attempts'].append({"problem_id": problem_id, "code_snapshot": user_code, "feedback": feedback_given})

    def generate_lesson_content(self, topic: str, difficulty: str = "beginner", num_exercises: int = 2) -> Dict[str, Any]:
        """
        Generates a full lesson, including markdown content and exercises.
        Updated prompt for more explicit JSON output and added robust parsing.
        """
        prompt = [
            f"You are an expert Python programming tutor. Generate a comprehensive Python lesson on the topic of '{topic}' "
            f"for a {difficulty} level programmer. The lesson should be engaging, clear, and include a main concept, "
            f"examples, and explanations. The lesson content should be under a 'lesson' key and structured with 'title', 'main_concept', and 'content' (a list of sections with 'heading' and 'text', and optionally 'example' for code snippets).",
            "After the lesson content, generate a section titled 'exercises:'. "
            f"For each exercise, provide:\n"
            f"1. A 'prompt' (the exercise description).\n"
            f"2. 'initial_code' (optional starting code for the user, usually empty or with a function signature).\n"
            f"3. 'check_function' (a Python function string named 'check_result(user_code, expected_output)' that "
            f"evaluates the user's code. This function **MUST NOT define functions or classes that the user is expected to implement.** " # Added this line
            f"Instead, it should `exec(user_code, user_exec_globals)` to make the user's definitions available. " # Clarified this
            f"It **MUST return a dictionary with keys 'passed' (bool), 'score' (float 0.0-1.0), and 'message' (str)**. "
            f"It should print messages indicating success or failure to the standard output. It "
            f"must use `exec` with a controlled `globals()` "
            f"to run user_code and then assert conditions. Be very careful to make this function robust and self-contained. It should handle imports like `math` if needed.\n"
            f"**REQUIRED `check_function` return structure:** `{{'passed': True, 'score': 1.0, 'message': 'Correct solution!'}}`\n"
            f"**Example `check_function` structure (pay close attention to the return type and how `user_code` is executed):**\n"
            f"```python\n"
            f"import io, sys, math # Example imports needed by check function or user code\n"
            f"def check_result(user_code, expected_output):\n"
            f"    captured_output = io.StringIO()\n"
            f"    original_stdout = sys.stdout\n"
            f"    sys.stdout = captured_output\n"
            f"    try:\n"
            f"        user_exec_globals = {{'__builtins__': __builtins__, 'math': math}}\n"
            f"        # Execute user's code to make their functions/variables available in user_exec_globals\n" # Clarified comment
            f"        exec(user_code, user_exec_globals)\n"
            f"        sys.stdout = original_stdout # Restore stdout after user code runs\n"
            f"        user_output = captured_output.getvalue().strip()\n"
            f"        \n"
            f"        # Example: Test if a user-defined 'add_numbers' function works\n"
            f"        # Check if the function exists and produces the correct result\n"
            f"        if 'divide' in user_exec_globals and callable(user_exec_globals['divide']):\n" # Changed example to 'divide'
            f"            # Test normal division\n"
            f"            result_normal = user_exec_globals['divide'](10, 2)\n"
            f"            if result_normal != 5.0:\n"
            f"                return {{'passed': False, 'score': 0.5, 'message': 'Test failed: Incorrect division result for 10/2.'}}\n"
            f"            \n"
            f"            # Test zero division handling (assuming it prints to stdout based on expected_output)\n"
            f"            temp_output_capture = io.StringIO()\n" # Temporarily capture output for this specific test
            f"            sys.stdout = temp_output_capture\n"
            f"            user_exec_globals['divide'](10, 0) # Call with zero to trigger error handling\n"
            f"            sys.stdout = original_stdout # Restore after specific test\n"
            f"            zero_div_output = temp_output_capture.getvalue().strip()\n"
            f"            if expected_output and expected_output in zero_div_output:\n" # Check against expected_output
            f"                return {{'passed': True, 'score': 1.0, 'message': 'All tests passed!'}}\n"
            f"            else:\n"
            f"                return {{'passed': False, 'score': 0.8, 'message': f'ZeroDivisionError handling failed. Expected \"{{expected_output}}\", got \"{{zero_div_output}}\".'}}\n"
            f"        else:\n"
            f"            return {{'passed': False, 'score': 0.0, 'message': 'Function `divide` not found or not callable in user code.'}}\n"
            f"    except SyntaxError as e:\n"
            f"        sys.stdout = original_stdout # Restore stdout before returning error\n"
            f"        return {{'passed': False, 'score': 0.1, 'message': f'Syntax Error in user code: {{e}}'}}\n"
            f"    except Exception as e:\n"
            f"        sys.stdout = original_stdout # Restore stdout before returning error\n"
            f"        return {{'passed': False, 'score': 0.0, 'message': f'Runtime Error in user code: {{e}}'}}\n"
            f"    finally:\n"
            f"        sys.stdout = original_stdout # Always restore stdout\n"
            f"```\n" # End of example
            f"4. 'expected_output' (the expected console output from a correct solution, if applicable). Use the actual string, not a description.\n",
            f"Generate {num_exercises} exercises. Format the *entire* output as a single JSON string. Do NOT include any text, preambles, or markdown formatting (like ```json) outside the JSON block. The JSON should start directly with '{{' and end with '}}'. Ensure all internal strings (especially for 'check_function') are properly JSON-escaped (e.g., newlines as \\n, double quotes as \\). The top-level keys should be 'lesson' and 'exercises'."
        ]

        generation_config = {"temperature": 0.9, "max_output_tokens": 2000}
        response_text = self._generate_content_with_error_handling(prompt, generation_config=generation_config)
        
        # --- Start of JSON Extraction and Parsing Logic ---
        parsed_data = {}
        error_message = ""

        # Log the raw AI response for debugging
        print(f"\n--- Raw AI Response for '{topic}' (start) ---")
        print(response_text)
        print(f"--- Raw AI Response for '{topic}' (end) ---\n")

        try:
            # Attempt 1: Direct parse after stripping common markdown fences
            cleaned_response_text = response_text.strip()
            
            # Remove leading/trailing markdown code block fences if they exist
            if cleaned_response_text.startswith("```json"):
                cleaned_response_text = cleaned_response_text[len("```json"):].strip()
            if cleaned_response_text.endswith("```"):
                cleaned_response_text = cleaned_response_text.rstrip("```").strip()

            # Attempt to find the actual JSON object (from first { to last })
            first_brace = cleaned_response_text.find('{')
            last_brace = cleaned_response_text.rfind('}')

            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_candidate = cleaned_response_text[first_brace : last_brace + 1]
                print(f"Attempting to parse extracted JSON candidate:\n{json_candidate[:500]}...")
                parsed_data = json.loads(json_candidate)
            else:
                raise ValueError("No valid JSON object delimiters found in the response.")

        except (json.JSONDecodeError, ValueError) as e:
            error_message = f"Error decoding JSON from AI (Attempt 1): {e}. Trying fallback regex."
            print(error_message)
            print(f"Problematic JSON candidate: {json_candidate[:500]}...") # Log what failed to parse

            # Attempt 2: Fallback regex search for anything resembling a JSON block
            json_match = re.search(r"```json\n(.*?)```", response_text, re.DOTALL)
            if json_match:
                try:
                    re_matched_json_string = json_match.group(1).strip()
                    print(f"Successfully extracted JSON via regex fallback:\n{re_matched_json_string[:500]}...")
                    parsed_data = json.loads(re_matched_json_string)
                    error_message = "" # Clear error if successful
                except json.JSONDecodeError as re_e:
                    error_message = f"Error decoding JSON from regex match fallback (Attempt 2): {re_e}"
                    print(error_message)
                    print(f"Regex matched string that failed to parse: {re_matched_json_string[:500]}...")
            else:
                error_message = f"Error: Could not find a parseable JSON structure in the AI response even with regex fallback."
                print(error_message)
        except Exception as e:
            error_message = f"Unexpected error during AI response processing: {e}"
            print(error_message)
            print(f"Traceback: {traceback.format_exc()}")
        
        # --- Post-processing and Error Return ---
        if not parsed_data or error_message:
            return {
                "lesson_content_markdown": f"Error: Could not generate lesson content. {error_message} Raw response (truncated): {response_text[:500]}...",
                "exercises": []
            }

# --- MODIFIED LOGIC HERE ---
        # The AI returns 'lesson' and 'exercises' as top-level keys.
        # We need to extract the actual markdown content from the 'lesson' dictionary.
        lesson_data = parsed_data.get('lesson', {})
        exercises_data = parsed_data.get('exercises', [])

        lesson_markdown_parts = []
        if 'title' in lesson_data:
            lesson_markdown_parts.append(f"# {lesson_data['title']}\n")
        elif 'lesson_title' in lesson_data: # Handle the 'lesson_title' case if it occurs
            lesson_markdown_parts.append(f"# {lesson_data['lesson_title']}\n")
        
        if 'main_concept' in lesson_data:
            lesson_markdown_parts.append(f"**Main Concept:** {lesson_data['main_concept']}\n")

        if 'content' in lesson_data and isinstance(lesson_data['content'], list):
            for section in lesson_data['content']:
                if 'heading' in section:
                    lesson_markdown_parts.append(f"## {section['heading']}\n")
                if 'text' in section:
                    lesson_markdown_parts.append(f"{section['text']}\n")
                if 'example' in section:
                    lesson_markdown_parts.append(f"```python\n{section['example']}\n```\n")
        
        # Also, check for other structures in the 'lesson' key (e.g., 'if_statements', 'loops')
        # and convert them to markdown if they exist.
        if 'if_statements' in lesson_data:
            lesson_markdown_parts.append("## If Statements\n")
            lesson_markdown_parts.append(f"{lesson_data['if_statements'].get('description', '')}\n")
            if 'example' in lesson_data['if_statements']:
                lesson_markdown_parts.append(f"```python\n{lesson_data['if_statements']['example']}\n```\n")

        if 'loops' in lesson_data:
            lesson_markdown_parts.append("## Loops\n")
            lesson_markdown_parts.append(f"{lesson_data['loops'].get('description', '')}\n")
            if 'for_loop_example' in lesson_data['loops']:
                lesson_markdown_parts.append(f"### For Loop Example\n")
                lesson_markdown_parts.append(f"```python\n{lesson_data['loops']['for_loop_example']}\n```\n")
            if 'while_loop_example' in lesson_data['loops']:
                lesson_markdown_parts.append(f"### While Loop Example\n")
                lesson_markdown_parts.append(f"```python\n{lesson_data['loops']['while_loop_example']}\n```\n")
        
        # Continue for other keys like 'nested_loops', 'break_and_continue' if they are present in the lesson structure
        if 'nested_loops' in lesson_data:
            lesson_markdown_parts.append("## Nested Loops\n")
            lesson_markdown_parts.append(f"{lesson_data['nested_loops'].get('description', '')}\n")
            if 'example' in lesson_data['nested_loops']:
                lesson_markdown_parts.append(f"```python\n{lesson_data['nested_loops']['example']}\n```\n")
        
        if 'break_and_continue' in lesson_data:
            lesson_markdown_parts.append("## Break and Continue\n")
            lesson_markdown_parts.append(f"{lesson_data['break_and_continue'].get('description', '')}\n")
            if 'example' in lesson_data['break_and_continue']:
                lesson_markdown_parts.append(f"```python\n{lesson_data['break_and_continue']['example']}\n```\n")


        lesson_content_markdown = "".join(lesson_markdown_parts)

        # Clean up exercises data
        for exercise in exercises_data:
            if 'prompt' in exercise:
                exercise['prompt'] = exercise['prompt'].replace('\u00A0', ' ')
            if 'initial_code' in exercise:
                exercise['initial_code'] = exercise['initial_code'].replace('\u00A0', ' ')
            if 'check_function' in exercise:
                check_func = exercise['check_function']
                check_func = check_func.replace('\u00A0', ' ') # Replace non-breaking space
                exercise['check_function'] = check_func
        
        return {
            "lesson_content_markdown": lesson_content_markdown,
            "exercises": exercises_data
        }

    def generate_exercise_solution_check(self, problem_description: str, expected_output: str, initial_user_code: str) -> str:
        """
        Generates a `check_function` string that evaluates pre-executed user code.
        The generated `check_result` function will receive the user's captured stdout
        and their final global variables, not the raw user code string.
        """
        prompt = [
            f"You are a Python programming expert. Given the following problem description and expected output, "
            f"generate a Python function string named 'check_result(user_output, user_globals)' that evaluates if the "
            f"user's code (which has ALREADY BEEN EXECUTED) correctly solves the problem. "
            f"THE FUNCTION MUST RETURN A DICTIONARY in the exact format "
            f"{{'passed': True/False, 'score': float, 'message': 'string explaining result'}}. "
            f"The score should be 1.0 for perfect, 0.0 for complete failure, or a float in between for partial success.\n"
            f"The `check_result` function will receive:\n"
            f" - `user_output`: A string containing all captured `stdout` from the user's executed code.\n"
            f" - `user_globals`: A dictionary containing the global variables and functions defined by the user's executed code.\n"
            f"**IMPORTANT:** Your `check_result` function MUST NOT attempt to `exec()` the user's code itself. It should only evaluate the `user_output` and `user_globals` dictionaries.\n\n" # <-- CRITICAL CHANGE HERE
            f"The function should:\n"
            f"1. Perform robust checks against the `expected_output` and the `user_output`.\n"
            f"2. If the problem required a specific function or variable to be defined, check for its existence and correctness within `user_globals`.\n"
            f"3. Be robust and handle cases where `user_output` might be empty or `user_globals` might lack expected keys.\n"
            f"4. Ensure necessary modules like `math`, `random`, etc., are imported within `check_result` if your check logic requires them (e.g., if you need `math.isclose`).\n\n"
            f"**Example `check_result` function structure and return format:**\n"
            f"```python\n"
            f"import math # Example: if your check logic needs math functions\n"
            f"def check_result(user_output, user_globals):\n" # <--- Updated signature
            f"    # Ensure outputs are stripped for fair comparison\n"
            f"    user_output_stripped = user_output.strip()\n"
            f"    expected_output_stripped = \"\" # Placeholder; your main.py will pass the real one\n\n" # Will be replaced by main.py
            f"    # Check 1: Basic output comparison\n"
            f"    if user_output_stripped == expected_output_stripped:\n"
            f"        return {{'passed': True, 'score': 1.0, 'message': 'Output matches exactly.'}}\n"
            f"    elif expected_output_stripped and expected_output_stripped in user_output_stripped:\n"
            f"        return {{'passed': True, 'score': 0.8, 'message': 'Output contains expected text, but may have extra.'}}\n"
            f"    elif expected_output_stripped:\n"
            f"        return {{'passed': False, 'score': 0.5, 'message': f'Output mismatch. Expected \"{{expected_output_stripped}}\", got \"{{user_output_stripped}}\".'}}\n"
            f"    \n"
            f"    # Check 2: Example of checking user-defined functions/variables in user_globals\n"
            f"    # if problem requires defining a function 'add'\n"
            f"    if 'add' in user_globals and callable(user_globals['add']):\n"
            f"        try:\n"
            f"            # Test the user's function\n"
            f"            if user_globals['add'](2, 3) == 5 and user_globals['add'](-1, 1) == 0:\n"
            f"                return {{'passed': True, 'score': 1.0, 'message': 'Output correct and function behaves as expected.'}}\n"
            f"            else:\n"
            f"                return {{'passed': False, 'score': 0.7, 'message': 'Output correct, but user-defined function has issues.'}}\n"
            f"        except Exception as e:\n"
            f"            return {{'passed': False, 'score': 0.6, 'message': f'User-defined function crashed: {{e}}'}}\n"
            f"    \n"
            f"    # Default case if no specific checks are met (e.g., just execute and no specific output)\n"
            f"    return {{'passed': False, 'score': 0.0, 'message': 'No specific checks passed or unexpected behavior.'}}\n"
            f"```\n\n" # End of example
            f"Problem Description: {problem_description}\n"
            f"Expected Output: {expected_output}\n" # This is now just information for the AI to understand the context
            f"Generate ONLY the Python function string, no extra text or markdown outside the function. Ensure all internal strings are properly escaped for JSON if this function string will be embedded in JSON later."
        ]
        # Lower temperature for more factual/direct generation of code
        generation_config = {"temperature": 0.5, "max_output_tokens": 800}
        # In this method, you might not pass user_code directly, as it's just for context for the AI
        return self._generate_content_with_error_handling(prompt, generation_config=generation_config)

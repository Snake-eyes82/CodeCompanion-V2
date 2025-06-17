import sys
import os
import json
import logging
import io
import builtins
import traceback
from functools import partial

# PySide6 imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QSplitter, QListWidget, QListWidgetItem,
    QLabel, QMessageBox, QDialog, QComboBox, QLineEdit, QInputDialog,
    QStatusBar, QMenuBar, QMenu, QSizePolicy
)
from PySide6.QtCore import Qt, QSettings, QSize
from PySide6.QtGui import QFont, QAction, QIntValidator, QColor, QTextCharFormat, QSyntaxHighlighter, QTextDocument

# Local application imports
from ai_agent import SelfImprovingAgent # Assuming ai_agent.py exists and defines AIAgent


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PythonLearningTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CodeCompanion V2 - Interactive Python Learning")
        self.setGeometry(100, 100, 1200, 800) # Increased initial window size

        # Initialize core data structures
        self.lessons_data = []
        self.current_lesson_index = -1
        self.current_exercise_index = -1
        self.exercise_attempts = {} # Stores attempts for exercises: {(lesson_idx, exercise_idx): num_attempts}

        # Paths for lessons and configurations
        self.lessons_dir = "lessons"
        self.lessons_generated_dir = os.path.join(self.lessons_dir, "generated")
        self.lessons_json_path = os.path.join(self.lessons_generated_dir, "lessons.json")

        # Ensure lesson directories exist
        os.makedirs(self.lessons_dir, exist_ok=True)
        os.makedirs(self.lessons_generated_dir, exist_ok=True)

        # AI Agent and settings initialization
        self.ai_agent = None
        self.google_api_key = "" # Will be loaded from settings/env
        self.ai_enabled = True # User toggle for AI features (can be overridden by global_ai_disabled)
        self.ai_enabled_globally = False # True only if API key is set and AI agent initialized
        self.ai_model_name = "gemini-1.5-flash"
        self.max_ai_tokens = 1000

        # Application settings for persistence
        self.settings = QSettings("CodeCompanion", "PythonLearningTool")
        self.current_theme = "Dark" # Default theme
        self.code_font_size = 10 # Default font size

        # UI Initialization and layout setup
        self.init_ui()
        self.load_settings() # Load saved settings and initialize AI agent based on key

        # Initial content loading
        self.load_lessons(self.lessons_json_path ) # Load lessons from JSON or generate if not found


    def init_ui(self):
        """Initializes the main user interface components and layout."""
        # Central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Create a splitter for lesson list and lesson content
        self.top_splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.top_splitter)

        # Lesson List on the left
        self.lesson_list_widget = QListWidget()
        self.lesson_list_widget.setObjectName("lessonList") # Object name for QSS styling
        self.lesson_list_widget.setMaximumWidth(250) # Set a fixed maximum width
        self.lesson_list_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding) # Fixed width, expanding height
        self.top_splitter.addWidget(self.lesson_list_widget)

        # Right side of the top splitter (Lesson Content and Code Editor)
        self.right_top_container = QWidget()
        self.right_top_layout = QVBoxLayout(self.right_top_container)
        self.top_splitter.addWidget(self.right_top_container)

        # Lesson Content Display
        self.lesson_content_text_edit = QTextEdit()
        self.lesson_content_text_edit.setObjectName("lessonContent") # Object name for QSS styling
        self.lesson_content_text_edit.setReadOnly(True)
        self.right_top_layout.addWidget(self.lesson_content_text_edit)

        # Code Editor and Output Splitter
        self.bottom_splitter = QSplitter(Qt.Vertical)
        self.right_top_layout.addWidget(self.bottom_splitter)

        # Code Editor
        self.code_editor = QTextEdit()
        self.code_editor.setObjectName("codeEditor") # Object name for QSS styling
        self.bottom_splitter.addWidget(self.code_editor)

        # Output Text
        self.output_text_edit = QTextEdit()
        self.output_text_edit.setObjectName("outputText") # Object name for QSS styling
        self.output_text_edit.setReadOnly(True)
        self.bottom_splitter.addWidget(self.output_text_edit)

        # Set initial sizes for splitters (adjust as needed)
        self.top_splitter.setSizes([200, 800]) # Initial widths for lesson list and content area
        self.bottom_splitter.setSizes([500, 300]) # Initial heights for code editor and output

        # Controls for exercises and lessons
        self.exercise_navigation_layout = QHBoxLayout()
        self.prev_exercise_button = QPushButton("Previous Exercise")
        self.next_exercise_button = QPushButton("Next Exercise")
        self.run_code_button = QPushButton("Run Code")
        self.check_answer_button = QPushButton("Check Answer (AI)")

        self.exercise_navigation_layout.addWidget(self.prev_exercise_button)
        self.exercise_navigation_layout.addWidget(self.next_exercise_button)
        self.exercise_navigation_layout.addStretch(1) # Pushes buttons to the left
        self.exercise_navigation_layout.addWidget(self.run_code_button)
        self.exercise_navigation_layout.addWidget(self.check_answer_button)
        self.main_layout.addLayout(self.exercise_navigation_layout)

        self.lesson_navigation_layout = QHBoxLayout()
        self.prev_lesson_button = QPushButton("Previous Lesson")
        self.next_lesson_button = QPushButton("Next Lesson")
        self.lesson_navigation_layout.addWidget(self.prev_lesson_button)
        self.lesson_navigation_layout.addWidget(self.next_lesson_button)
        self.lesson_navigation_layout.addStretch(1)
        self.main_layout.addLayout(self.lesson_navigation_layout)

        # AI Question Input
        self.ai_question_input = QTextEdit()
        self.ai_question_input.setObjectName("aiQuestionInput")
        self.ai_question_input.setPlaceholderText("Ask the AI a general question or for help...")
        self.ai_question_input.setFixedHeight(50) # Fixed height for input
        self.main_layout.addWidget(self.ai_question_input)

        self.ai_buttons_layout = QHBoxLayout()
        self.ask_ai_general_help_button = QPushButton("Ask The AI")
        self.ai_buttons_layout.addStretch(1)
        self.ai_buttons_layout.addWidget(self.ask_ai_general_help_button)
        self.main_layout.addLayout(self.ai_buttons_layout)


        # Status Bar
        self.status_bar = self.statusBar()
        self.ai_status_label = QLabel("AI Status: Initializing...")
        self.status_bar.addWidget(self.ai_status_label)
        self.score_label = QLabel("Score: 0") # Placeholder for score
        self.status_bar.addPermanentWidget(self.score_label) # Align to the right

        # Menu Bar
        self.create_menu_bar()

        # Connect signals and slots
        self.lesson_list_widget.itemClicked.connect(self.on_lesson_selected)
        self.run_code_button.clicked.connect(self.run_user_code)
        self.check_answer_button.clicked.connect(self.check_answer_with_ai)
        self.prev_exercise_button.clicked.connect(self.prev_exercise)
        self.next_exercise_button.clicked.connect(self.next_exercise)
        self.prev_lesson_button.clicked.connect(self.prev_lesson)
        self.next_lesson_button.clicked.connect(self.next_lesson)
        self.ask_ai_general_help_button.clicked.connect(self.ask_ai_general_help)

        # Apply initial theme and font size
        self.apply_theme(self.current_theme) # Applies dark/light theme based on settings
        self.apply_font_size(self.code_font_size) # Applies font size to text areas

        # Update button states
        self.update_navigation_buttons()
        self.update_exercise_buttons_state()


    def create_menu_bar(self):
        """Creates the application's menu bar with various actions."""
        menu_bar = self.menuBar()

        # File Menu
        file_menu = menu_bar.addMenu("&File")
        exit_action = QAction("E&xit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Settings Menu
        settings_menu = menu_bar.addMenu("&Settings")

        # Theme Sub-menu
        theme_menu = settings_menu.addMenu("&Theme")
        dark_theme_action = QAction("Dark Theme", self)
        dark_theme_action.triggered.connect(lambda: self.apply_theme("Dark"))
        theme_menu.addAction(dark_theme_action)

        light_theme_action = QAction("Light Theme", self)
        light_theme_action.triggered.connect(lambda: self.apply_theme("Light"))
        theme_menu.addAction(light_theme_action)

        # Font Size Action
        font_size_action = QAction("Font Size...", self)
        font_size_action.triggered.connect(self.show_font_size_dialog)
        settings_menu.addAction(font_size_action)

        # AI Model Settings Action
        ai_model_settings_action = QAction("AI Model Settings...", self)
        ai_model_settings_action.triggered.connect(self.show_ai_model_settings_dialog)
        settings_menu.addAction(ai_model_settings_action)

        # Toggle AI Features Action
        self.toggle_ai_action = QAction("Enable AI Features", self)
        self.toggle_ai_action.setCheckable(True)
        self.toggle_ai_action.setChecked(self.ai_enabled) # Set initial state based on self.ai_enabled
        self.toggle_ai_action.triggered.connect(self.toggle_ai_features)
        settings_menu.addAction(self.toggle_ai_action)


        # Help Menu
        help_menu = menu_bar.addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

    def load_lessons(self, file_path): # Add a comma here!
        """
        Loads lessons from lessons.json. If the file doesn't exist or is empty,
        it attempts to generate initial lessons using the AI.
        """
        self.lessons_data = [] # Clear existing lessons
        self.lesson_list_widget.clear() # Clear lesson list UI

        # IMPORTANT: The logic below still uses 'self.lessons_json_path'
        # but your generate_initial_lessons_with_ai is calling load_lessons(lessons_file_path)
        # This means you intended to use 'file_path' argument to load.
        # You need to decide if load_lessons always uses self.lessons_json_path
        # or if it uses the passed file_path.
        # Based on the error, you were passing lessons_file_path, so load_lessons should use it.

        # I recommend making load_lessons primarily use the passed file_path
        # and letting the caller (like generate_initial_lessons_with_ai or __init__)
        # decide which file to load.

        # --- REVISED LOGIC FOR load_lessons to use the passed file_path ---
        if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
            logging.info(f"'{file_path}' not found or is empty. Attempting to generate initial lessons.")
            self.output_text_edit.append("AI: No lessons found. Attempting to generate initial lessons. This may take a moment...")
            if QApplication.instance():
                QApplication.instance().processEvents() # Update UI

            self.generate_initial_lessons_with_ai()
            # After generation, generate_initial_lessons_with_ai *already calls*
            # self.load_lessons(lessons_file_path) at its end. So, this branch
            # should ideally not proceed to try loading the file itself, but rely
            # on the successful generation path to call load_lessons again.
            # However, for simplicity and to avoid recursive calls, we'll keep the
            # current structure but modify the loading path below.
            # A cleaner approach would be for generate_initial_lessons_with_ai
            # to just save the file, and then the main __init__ calls load_lessons.

            # For now, let's ensure the initial call to load_lessons from __init__
            # passes self.lessons_json_path, and then the call from
            # generate_initial_lessons_with_ai passes lessons_file_path.
            # This method should then use the 'file_path' argument for loading.
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as f: # Use file_path here
                    self.lessons_data = json.load(f)
                logging.info(f"Successfully loaded lessons from '{file_path}'.") # Use file_path here
                self.output_text_edit.append(f"Loaded {len(self.lessons_data)} lessons.")
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding lessons from '{file_path}': {e}", exc_info=True) # Use file_path
                self.output_text_edit.append(f"Error loading lessons from '{file_path}': {e}. Attempting to generate new lessons.") # Use file_path
                self.generate_initial_lessons_with_ai() # This will generate and then call load_lessons again
            except Exception as e:
                logging.error(f"An unexpected error occurred while loading lessons from '{file_path}': {e}", exc_info=True) # Use file_path
                self.output_text_edit.append(f"An unexpected error occurred: {e}. Attempting to generate new lessons.")
                self.generate_initial_lessons_with_ai()

        self.update_lesson_list_widget()
        if self.lessons_data:
            self.load_lesson(0) # Load the first lesson by default (this loads an individual exercise)
        else:
            self.lesson_content_text_edit.setPlainText("No lessons available. Please check AI settings or generate lessons.")
            self.code_editor.setPlainText("")
            self.output_text_edit.append("No lessons could be loaded or generated.")

    def generate_initial_lessons_with_ai(self):
        """Generates an initial set of Python lessons and exercises using the AI."""
        if not self.ai_enabled_globally or self.ai_agent is None or self.ai_agent.chat is None:
            logging.warning("AI is not enabled or ready. Cannot generate initial lessons.")
            self.output_text_edit.append("\nAI Error: Cannot generate initial lessons, AI is not enabled or ready.")
            QMessageBox.warning(self, "AI Not Ready", "AI is not enabled or ready. Cannot generate initial lessons.")
            return

        logging.info("Attempting to generate initial lessons.")
        initial_lesson_prompt = """
        Generate a comprehensive JSON array of introductory Python lessons and exercises, suitable for a beginner.
        Each lesson should have a 'title' and an 'exercises' array.
        Each exercise should have:
        - 'title': A concise title for the exercise.
        - 'prompt': Clear instructions for the user.
        - 'initial_code': Starting code for the user to complete. Use placeholders like `...` or leave blanks.
        - 'expected_output': The exact string expected from the code's output.
        - 'check_function': A Python function (as a string) named 'check_solution' that takes 'user_code', 'user_output', and 'expected_output' as arguments. This function MUST return a JSON string containing two keys:
            - 'is_correct': A boolean (True/False) indicating if the solution is correct.
            - 'feedback': A string providing specific, concise feedback related to the correctness or issues found by the checker.
            For simple 'print' exercises, the `check_solution` should primarily focus on comparing 'user_output' to 'expected_output' and detecting basic structural elements like the 'print' statement.

        Ensure the JSON is perfectly valid and complete, without any introductory or concluding text outside the JSON structure.

        Example Structure:
        [
            {
                "title": "Lesson 1: Introduction to Python",
                "exercises": [
                    {
                        "title": "Hello World",
                        "prompt": "Write a Python program that prints 'Hello, World!' to the console.",
                        "initial_code": "print(...)",
                        "expected_output": "Hello, World!",
                        "check_function": "import json\\n\\ndef check_solution(user_code, user_output, expected_output):\\n    is_correct = user_output.strip() == expected_output.strip() and 'print' in user_code\\n    if is_correct:\\n        return json.dumps({\"is_correct\": True, \"feedback\": \"Output matches and 'print' function is used.\"})\\n    else:\\n        feedback_msg = \"Output does not match expected. Make sure you print 'Hello, World!' exactly.\"\\n        if 'print' not in user_code:\\n            feedback_msg += \"\\\\nEnsure you are using the 'print' function.\"\\n        return json.dumps({\"is_correct\": False, \"feedback\": feedback_msg.strip()})"
                    },
                    {
                        "title": "Print a Specific Name",
                        "prompt": "Write a program that prints the name 'Alice' to the console. Make sure the output is exactly 'Alice'.",
                        "initial_code": "name = \"...\"\\nprint(name)",
                        "expected_output": "Alice",
                        "check_function": "import json\\n\\ndef check_solution(user_code, user_output, expected_output):\\n    is_correct = user_output.strip() == expected_output.strip() and 'print' in user_code\\n    if is_correct:\\n        return json.dumps({\"is_correct\": True, \"feedback\": \"Great job! The output matches the expected name.\"})\\n    else:\\n        feedback_msg = \"The output does not match 'Alice'. Check your spelling, capitalization, and make sure you are only printing the name.\"\\n        if 'print' not in user_code:\\n            feedback_msg += \"\\\\nRemember to use the 'print()' function.\"\\n        return json.dumps({\"is_correct\": False, \"feedback\": feedback_msg.strip()})"
                    }
                ]
            }
        ]
        """
        lesson_plan_response = "" # Initialize here
        try:
            response_obj = self.ai_agent.chat.send_message(initial_lesson_prompt)
            
            if response_obj.parts:
                lesson_plan_response = response_obj.parts[0].text
            else:
                lesson_plan_response = response_obj.text
            
            # --- MODIFIED: Strip both the "AI Response:" (if it somehow returns) AND the Markdown code block markers ---
            json_string_to_parse = lesson_plan_response.strip() # Start by stripping general whitespace

            # If the AI adds "AI Response:\n" (though now it seems to be adding ```json)
            prefix_ai_response = "AI Response:\n"
            if json_string_to_parse.startswith(prefix_ai_response):
                json_string_to_parse = json_string_to_parse[len(prefix_ai_response):].strip()

            # --- NEW: Strip markdown code block fences ---
            if json_string_to_parse.startswith("```json"):
                json_string_to_parse = json_string_to_parse[len("```json"):].strip()
            
            if json_string_to_parse.endswith("```"):
                json_string_to_parse = json_string_to_parse[:-len("```")].strip()
            # --- END NEW ---

            # Parse the JSON response
            raw_lessons = json.loads(json_string_to_parse) # Use the cleaned string here

            # Validate the structure (optional but recommended for robustness)
            if not isinstance(raw_lessons, list):
                raise ValueError("AI response is not a JSON array.")
            for lesson in raw_lessons:
                if not all(k in lesson for k in ["title", "exercises"]):
                    raise ValueError("Lesson missing 'title' or 'exercises'.")
                if not isinstance(lesson["exercises"], list):
                    raise ValueError("Exercises is not a JSON array.")
                for exercise in lesson["exercises"]:
                    if not all(k in exercise for k in ["title", "prompt", "initial_code", "expected_output"]):
                        raise ValueError("Exercise missing required fields.")

            # Save the lessons
            lessons_dir = os.path.join(os.path.dirname(__file__), "lessons", "generated") # More robust path
            os.makedirs(lessons_dir, exist_ok=True)
            lessons_file_path = os.path.join(lessons_dir, 'lessons.json')
            with open(lessons_file_path, 'w', encoding='utf-8') as f:
                json.dump(raw_lessons, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Initial lessons generated and saved to '{lessons_file_path}'.")
            self.load_lessons(lessons_file_path) # Load the newly generated lessons

        except json.JSONDecodeError as e:
            # Keep the detailed error logging here, it's very useful
            logging.error(f"Failed to decode initial lesson plan from AI: {e}. Raw response: {lesson_plan_response}", exc_info=True)
            self.output_text_edit.append(f"\nAI Error: Failed to generate lessons. The AI's response was not valid JSON.")
            QMessageBox.critical(self, "AI Generation Error", f"Failed to generate initial lessons due to invalid AI response. Please check the console for details: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred during AI lesson generation: {e}", exc_info=True)
            self.output_text_edit.append(f"\nAI Error: An unexpected error occurred during lesson generation.")
            QMessageBox.critical(self, "AI Generation Error", f"An unexpected error occurred during AI lesson generation. Please check the console for details: {e}")
    
    def load_lesson(self, index):
        """
        Loads and displays the lesson content and the first exercise for the given index.
        """
        if not self.lessons_data:
            self.lesson_content_text_edit.setPlainText("No lessons loaded.")
            self.code_editor.setPlainText("")
            self.current_lesson_index = -1
            self.current_exercise_index = -1
            self.update_navigation_buttons()
            self.update_exercise_buttons_state()
            return

        if 0 <= index < len(self.lessons_data):
            self.current_lesson_index = index
            lesson = self.lessons_data[self.current_lesson_index]

            # Display lesson title and exercises in lesson_content_text_edit
            lesson_title = lesson.get('title', f"Lesson {index + 1}")
            content_display = f"# {lesson_title}\n\n"
            content_display += "## Exercises:\n"
            for i, exercise in enumerate(lesson.get('exercises', [])):
                content_display += f"- {i+1}. {exercise.get('title', f'Exercise {i+1}')}\n"
            self.lesson_content_text_edit.setMarkdown(content_display) # Use Markdown for formatting

            self.output_text_edit.clear() # Clear output for new lesson

            # Automatically load the first exercise
            if lesson.get('exercises'):
                self.display_exercise(0) # Display the first exercise
            else:
                self.output_text_edit.append("No exercises for this lesson.")
                self.code_editor.setPlainText("")
                self.current_exercise_index = -1

            # Update lesson list selection
            self.lesson_list_widget.setCurrentRow(self.current_lesson_index)
            logging.info(f"Loaded lesson: {lesson_title}")
        else:
            logging.warning(f"Attempted to load invalid lesson index: {index}")
            self.output_text_edit.append(f"Error: Invalid lesson index {index}.")

        self.update_navigation_buttons()
        self.update_exercise_buttons_state()


    def on_lesson_selected(self, item):
        """Handles selection of a lesson from the QListWidget."""
        index = self.lesson_list_widget.row(item)
        self.load_lesson(index)

    def display_exercise(self, index):
        """Displays the prompt and initial code for a specific exercise."""
        if self.current_lesson_index == -1 or not self.lessons_data:
            return

        lesson = self.lessons_data[self.current_lesson_index]
        exercises = lesson.get('exercises', [])

        if 0 <= index < len(exercises):
            self.current_exercise_index = index
            exercise = exercises[self.current_exercise_index]

            self.lesson_content_text_edit.setMarkdown(
                f"# {lesson.get('title', 'Lesson')}\n\n"
                f"## Exercise {index + 1}: {exercise.get('title', 'Untitled Exercise')}\n\n"
                f"{exercise.get('prompt', 'No prompt provided.')}"
            )
            self.code_editor.setPlainText(exercise.get('initial_code', '# Write your code here'))
            self.output_text_edit.clear()
            self.output_text_edit.append(f"--- Exercise {index + 1} Loaded ---")
            logging.info(f"Displayed exercise {index + 1} for lesson {self.current_lesson_index + 1}.")
            # Reset attempts for new exercise
            self.exercise_attempts[(self.current_lesson_index, self.current_exercise_index)] = 0
        else:
            self.output_text_edit.append("All exercises for this lesson completed!")
            self.code_editor.setPlainText("# Lesson Completed!")
            self.current_exercise_index = -1 # Indicate no active exercise
            if hasattr(self, 'check_answer_button'):
                self.check_answer_button.setEnabled(False)

        self.update_exercise_buttons_state()

    def check_answer_with_ai(self):
        """
        Submits user's code to the AI for checking against an expected output
        or an AI-generated check function.
        """
        if not self.ai_enabled or not self.ai_agent or self.ai_agent.api_status != "READY":
            self.output_text_edit.append("\nAI features are currently disabled or not ready. Cannot check answer.")
            return
        if self.current_lesson_index == -1 or self.current_exercise_index == -1:
            self.output_text_edit.append("\nNo active exercise to check.")
            return

        lesson = self.lessons_data[self.current_lesson_index]
        exercise = lesson.get('exercises', [])[self.current_exercise_index]
        
        user_code = self.code_editor.toPlainText()
        problem_description = exercise.get('prompt', 'N/A')
        expected_output = exercise.get('expected_output', '')
        ai_check_function_str = exercise.get('check_function', "")

        self.output_text_edit.append("\nAI: Checking your answer... Please wait.")
        if QApplication.instance():
            QApplication.instance().processEvents()

        # Increment attempt counter
        current_attempts = self.exercise_attempts.get((self.current_lesson_index, self.current_exercise_index), 0)
        self.exercise_attempts[(self.current_lesson_index, self.current_exercise_index)] = current_attempts + 1

        try:
            feedback_response = None
            if ai_check_function_str and "def check_solution(" in ai_check_function_str:
                # Attempt to use the AI-generated checker function
                try:
                    checker_globals = {}
                    exec(ai_check_function_str, checker_globals)
                    # --- MODIFIED: Get "check_solution" instead of "check_result" ---
                    check_result_func = checker_globals.get("check_solution")

                    if check_result_func:
                        # The check_result_func (now check_solution) needs to accept user_code, user_output, expected_output
                        # Based on your previous prompt: "check_solution(user_code, user_output, expected_output)"
                        # But your current call is check_result_func(user_code, expected_output)
                        # We need to *first* run the user_code to get user_output before passing to check_solution
                        
                        # --- NEW: Run user code to get user_output ---
                        self.output_text_edit.append("\n--- Running User Code ---")
                        captured_user_output = ""
                        try:
                            import sys
                            from io import StringIO
                            old_stdout = sys.stdout
                            redirected_output = StringIO()
                            sys.stdout = redirected_output

                            exec(user_code) # Execute the user's code

                            captured_user_output = redirected_output.getvalue().strip()
                            sys.stdout = old_stdout # Restore stdout

                            self.output_text_edit.append(f"Output:\n{captured_user_output}")
                            self.output_text_edit.append("--------------------------")

                        except Exception as e:
                            sys.stdout = old_stdout # Ensure stdout is restored
                            captured_user_output = f"Error during execution: {e}"
                            self.output_text_edit.append(f"Error executing code:\n{captured_user_output}")
                            self.output_text_edit.append("--------------------------")
                            logging.error(f"Error executing user code: {e}", exc_info=True)
                            # If user's code itself errors, no need to run AI check function, go straight to AI feedback
                            feedback_response = self.ai_agent.provide_feedback_on_code(
                                user_code=user_code,
                                problem_description=problem_description,
                                expected_output=expected_output,
                                previous_errors=f"User code execution error: {e}",
                                num_attempts=self.exercise_attempts[(self.current_lesson_index, self.current_exercise_index)]
                            )
                            # Display feedback and exit
                            if feedback_response:
                                self.output_text_edit.append(f"\n--- AI Feedback ---\n{feedback_response}")
                            return # IMPORTANT: Exit here if user code fails

                        # --- MODIFIED: Call check_solution with user_code, captured_user_output, expected_output ---
                        # The check_solution function should *not* execute user_code internally
                        # It should compare user_code, user_output to expected_output
                        check_output = check_result_func(user_code, captured_user_output, expected_output)
                        
                        # Assuming check_output is a JSON string like '{"is_correct": true, "feedback": "..."}'
                        check_output_dict = json.loads(check_output)
                        is_correct = check_output_dict.get('is_correct', False)
                        feedback = check_output_dict.get('feedback', 'No specific feedback from checker.')

                        if is_correct:
                            self.output_text_edit.append("\n--- Automated Check Result ---\n✅ Correct! " + feedback)
                            self.output_text_edit.append("Moving to the next exercise...")
                            self.next_exercise() # Automatically move to next exercise on success
                            return # Exit after successful check
                        else:
                            self.output_text_edit.append("\n--- Automated Check Result ---\n❌ Incorrect. " + feedback)
                            # Provide AI feedback based on the incorrect attempt
                            previous_errors = feedback # Use the checker's feedback as previous errors
                            feedback_response = self.ai_agent.provide_feedback_on_code(
                                user_code=user_code,
                                problem_description=problem_description,
                                expected_output=expected_output,
                                previous_errors=previous_errors,
                                num_attempts=self.exercise_attempts[(self.current_lesson_index, self.current_exercise_index)]
                            )
                    else:
                        raise ValueError("AI check function 'check_solution' not found after execution.") # MODIFIED name
                except Exception as e:
                    logging.warning(f"Error executing AI-generated check function: {e}. Falling back to general AI feedback.", exc_info=True)
                    self.output_text_edit.append(f"\nAI: Error with automated checker ({e}). Falling back to general AI feedback.")
                    # Fallback to general AI feedback if checker fails
                    previous_errors = f"Automated checker failed with error: {e}"
                    feedback_response = self.ai_agent.provide_feedback_on_code(
                        user_code=user_code,
                        problem_description=problem_description,
                        expected_output=expected_output,
                        previous_errors=previous_errors,
                        num_attempts=self.exercise_attempts[(self.current_lesson_index, self.current_exercise_index)]
                    )
            else: # This path is taken if ai_check_function_str is empty or doesn't contain "def check_solution("
                logging.info("No AI-generated check function available. Requesting general AI feedback.")
                self.output_text_edit.append("\nAI: No specific automated check available for this exercise. Requesting general feedback.")
                # Fallback to general AI feedback if no checker is present
                feedback_response = self.ai_agent.provide_feedback_on_code(
                    user_code=user_code,
                    problem_description=problem_description,
                    expected_output=expected_output,
                    previous_errors="No automated check was available, providing general guidance.",
                    num_attempts=self.exercise_attempts[(self.current_lesson_index, self.current_exercise_index)]
                )
            
            if feedback_response:
                self.output_text_edit.append(f"\n--- AI Feedback ---\n{feedback_response}")

        except Exception as e:
            logging.error(f"Error getting AI feedback: {e}", exc_info=True)
            self.output_text_edit.append(f"\nAI Error: Failed to get feedback: {e}")

    def get_ai_feedback_on_code(self, user_code: str, problem_description: str, expected_output: str):
        """
        Requests general AI feedback on user's code when no automated check is available.
        """
        logging.info("Requesting general AI feedback on user's code.")
        self.output_text_edit.append("\nAI: Generating general feedback on your code (no automated check available). Please wait...")
        if QApplication.instance():
            QApplication.instance().processEvents()

        try:
            feedback = self.ai_agent.provide_feedback_on_code(
                user_code=user_code,
                problem_description=problem_description,
                expected_output=expected_output,
                previous_errors="No automated check was available, providing general guidance.",
                num_attempts=1 # This is for general feedback, so num_attempts isn't critical but good to pass something
            )
            self.output_text_edit.append(f"\n--- AI General Feedback ---\n{feedback}")
        except Exception as e:
            logging.error(f"Error getting AI general feedback: {e}", exc_info=True)
            self.output_text_edit.append(f"\nAI Error: Failed to get general feedback: {e}")

    def update_exercise_prompt_in_display(self):
        """
        Updates the exercise prompt and code editor based on the current_exercise_index.
        This method is called after successfully completing an exercise or moving between them.
        """
        # This method now primarily serves as a wrapper to call display_exercise
        # or handle the "all exercises completed" case.
        if self.current_lesson_index == -1 or not self.lessons_data:
            self.code_editor.setPlainText("")
            return

        lesson = self.lessons_data[self.current_lesson_index]
        exercises = lesson.get('exercises', [])

        if not exercises:
            self.code_editor.setPlainText("")
            return

        if 0 <= self.current_exercise_index < len(exercises):
            self.display_exercise(self.current_exercise_index)
        else:
            self.output_text_edit.append("All exercises for this lesson completed!")
            self.code_editor.setPlainText("# Lesson Completed!")
            if hasattr(self, 'check_answer_button'):
                self.check_answer_button.setEnabled(False) # Disable check button when lesson is completed

        self.update_exercise_buttons_state()


    def update_navigation_buttons(self):
        """Logic to enable/disable previous/next lesson buttons."""
        if hasattr(self, 'prev_lesson_button') and hasattr(self, 'next_lesson_button'):
            if not self.lessons_data:
                self.prev_lesson_button.setEnabled(False)
                self.next_lesson_button.setEnabled(False)
            else:
                self.prev_lesson_button.setEnabled(self.current_lesson_index > 0)
                self.next_lesson_button.setEnabled(self.current_lesson_index < len(self.lessons_data) - 1)
        else:
            logging.debug("Lesson navigation buttons not initialized.")

    def update_exercise_buttons_state(self):
        """Logic to enable/disable prev/next exercise buttons and check answer button."""
        if hasattr(self, 'prev_exercise_button') and hasattr(self, 'next_exercise_button') and hasattr(self, 'check_answer_button'):
            if self.current_lesson_index == -1 or not self.lessons_data:
                self.prev_exercise_button.setEnabled(False)
                self.next_exercise_button.setEnabled(False)
                self.check_answer_button.setEnabled(False)
                self.run_code_button.setEnabled(False) # Disable run code if no active lesson/exercise
                return

            current_lesson = self.lessons_data[self.current_lesson_index]
            exercises = current_lesson.get('exercises', [])

            if not exercises or self.current_exercise_index == -1:
                self.prev_exercise_button.setEnabled(False)
                self.next_exercise_button.setEnabled(False)
                self.check_answer_button.setEnabled(False)
                self.run_code_button.setEnabled(False)
            else:
                self.prev_exercise_button.setEnabled(self.current_exercise_index > 0)
                self.next_exercise_button.setEnabled(self.current_exercise_index < len(exercises) - 1)
                # Enable check button only if AI is enabled and there's an active exercise
                self.check_answer_button.setEnabled(self.ai_enabled and self.ai_enabled_globally and self.ai_agent and self.ai_agent.api_status == "READY")
                self.run_code_button.setEnabled(True) # Always allow running code if exercise is active
        else:
            logging.debug("Exercise navigation or check buttons not initialized.")

    def next_exercise(self):
        """Moves to the next exercise in the current lesson."""
        if self.current_lesson_index != -1 and self.lessons_data:
            lesson = self.lessons_data[self.current_lesson_index]
            exercises = lesson.get('exercises', [])
            if self.current_exercise_index < len(exercises) - 1:
                self.display_exercise(self.current_exercise_index + 1)
            else:
                self.output_text_edit.append("Already on the last exercise of this lesson.")

    def prev_exercise(self):
        """Moves to the previous exercise in the current lesson."""
        if self.current_lesson_index != -1 and self.lessons_data:
            if self.current_exercise_index > 0:
                self.display_exercise(self.current_exercise_index - 1)
            else:
                self.output_text_edit.append("Already on the first exercise of this lesson.")

    def next_lesson(self):
        """Moves to the next lesson."""
        if self.lessons_data and self.current_lesson_index < len(self.lessons_data) - 1:
            self.load_lesson(self.current_lesson_index + 1)
        else:
            self.output_text_edit.append("Already on the last lesson.")

    def prev_lesson(self):
        """Moves to the previous lesson."""
        if self.current_lesson_index > 0:
            self.load_lesson(self.current_lesson_index - 1)
        else:
            self.output_text_edit.append("Already on the first lesson.")

    def update_score_label(self, text):
        """Updates the score display in the status bar."""
        self.score_label.setText(text)

    def update_lesson_list_widget(self):
        """
        Updates the lesson list widget with current lessons.
        Called after loading or generating lessons.
        """
        self.lesson_list_widget.clear()
        for i, lesson in enumerate(self.lessons_data):
            self.lesson_list_widget.addItem(lesson.get('title', f"Lesson {i+1}"))
        if 0 <= self.current_lesson_index < len(self.lessons_data):
            self.lesson_list_widget.setCurrentRow(self.current_lesson_index)


    def apply_syntax_highlighting(self):
        """
        Applies syntax highlighting to the code editor.
        (Placeholder - implementation requires a QSyntaxHighlighter subclass)
        """
        # A custom QSyntaxHighlighter class would be instantiated here
        # self.highlighter = PythonHighlighter(self.code_editor.document())
        pass

    def run_user_code(self):
        """
        Executes the user's code from the code editor in an isolated environment
        and displays its output.
        """
        self.output_text_edit.append("\n--- Running User Code ---")
        user_code = self.code_editor.toPlainText()
        output_buffer = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = output_buffer
        sys.stderr = output_buffer

        # Create a dictionary for the execution environment
        # Only allow a minimal set of builtins for safety
        safe_builtins = {name: getattr(builtins, name) for name in dir(builtins) if not name.startswith('__') and name not in ['open', 'eval', 'exec']}
        user_globals = {"__builtins__": safe_builtins}
        user_locals = {}

        try:
            # Execute the user code
            exec(user_code, user_globals, user_locals)
        except Exception as e:
            self.output_text_edit.append(f"Execution Error: {e}")
            self.output_text_edit.append(f"\nTraceback:\n{traceback.format_exc()}")
        finally:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            # Display captured output
            self.output_text_edit.append(f"\nOutput:\n{output_buffer.getvalue().strip()}")
            self.output_text_edit.append("\n--------------------------")

    def ask_ai_general_help(self):
        """
        Sends a general question from the user to the AI agent and displays the response.
        Provides context from the current lesson/exercise.
        """
        if not self.ai_enabled or not self.ai_agent or self.ai_agent.api_status != "READY":
            self.output_text_edit.append("\nAI features are currently disabled. Cannot ask for help.")
            return

        question = self.ai_question_input.toPlainText().strip()
        if not question:
            self.output_text_edit.append("\nPlease type your question in the 'Ask The AI' box before clicking the button.")
            return

        self.output_text_edit.append(f"\nUser: {question}")
        self.output_text_edit.append("\nAI: Generating response... Please wait.")
        if QApplication.instance():
            QApplication.instance().processEvents()

        try:
            # Get context from current lesson/exercise if available
            context_lesson_title = ""
            context_exercise_prompt = ""
            if self.current_lesson_index != -1 and self.lessons_data:
                lesson = self.lessons_data[self.current_lesson_index]
                context_lesson_title = lesson.get('title', 'N/A')
                exercises = lesson.get('exercises', [])
                if 0 <= self.current_exercise_index < len(exercises):
                    context_exercise_prompt = exercises[self.current_exercise_index].get('prompt', 'N/A')
            
            response = self.ai_agent.ask_general_question(
                question=question,
                current_lesson_title=context_lesson_title,
                current_exercise_prompt=context_exercise_prompt,
                user_code=self.code_editor.toPlainText()
            )
            self.output_text_edit.append(f"\n--- AI Response ---\n{response}")
            self.ai_question_input.clear() # Clear input after asking
        except Exception as e:
            logging.error(f"Error asking AI general help: {e}", exc_info=True)
            self.output_text_edit.append(f"\nAI Error: Failed to get general help: {e}")

    def load_settings(self):
        """
        Loads UI and AI preferences from QSettings.
        Initializes the AI agent if an API key is found.
        """
        self.current_theme = self.settings.value("current_theme", "Dark", str)
        self.code_font_size = self.settings.value("code_font_size", 10, int)
        self.ai_enabled = self.settings.value("ai_enabled", True, bool) # User toggle for AI features
        self.ai_model_name = self.settings.value("ai_model_name", "gemini-1.5-flash", str)
        self.max_ai_tokens = self.settings.value("max_ai_tokens", 1000, int)

        # Load API key from QSettings first
        loaded_api_key = self.settings.value("google_api_key", "", str)

        # If no key found in QSettings, try to load from the GOOGLE_API_KEY environment variable (e.g., from .env file)
        if not loaded_api_key:
            loaded_api_key = os.environ.get("GOOGLE_API_KEY", "")

        self.google_api_key = loaded_api_key # Store the determined key in the instance attribute

        logging.info("Settings loaded via QSettings.")
        self.apply_theme(self.current_theme)
        self.apply_font_size(self.code_font_size)

        # Initialize AI agent if a key was found (either from QSettings or .env)
        if self.google_api_key:
            logging.info("Attempting to initialize AI agent.")
            self.initialize_ai_agent(self.google_api_key) # Call the helper method to initialize AI
        else:
            logging.warning("No Google API Key found in settings or environment. AI features will be globally disabled.")
            self.ai_enabled_globally = False # Ensure AI is globally disabled if no key
            self.ai_agent = None # Make sure agent is None if no key

        # Set the initial checked state of the menu action
        self.toggle_ai_action.setChecked(self.ai_enabled)
        # Disable the toggle if AI is globally disabled (no key)
        self.toggle_ai_action.setEnabled(self.ai_enabled_globally)
        self.ask_ai_general_help_button.setEnabled(self.ai_enabled and self.ai_enabled_globally)
        self.check_answer_button.setEnabled(self.ai_enabled and self.ai_enabled_globally)

        self.update_ai_status_label() # Update AI status based on initial load

    def initialize_ai_agent(self, api_key: str):
        """Initializes the AI agent with the given API key and configured model/tokens."""
        if api_key:
            try:
                self.ai_agent = SelfImprovingAgent(api_key=api_key)
                #self.ai_agent = SelfImprovingAgent(api_key=api_key, model_name=self.ai_model_name, max_tokens=self.max_ai_tokens)
                # Perform a quick test call to verify API status (optional but good)
                # For simplicity, we'll just assume READY if instantiation succeeds
                self.ai_enabled_globally = True
                logging.info("AI agent initialized successfully.")
            except Exception as e:
                logging.error(f"Failed to initialize AI agent: {e}", exc_info=True)
                self.output_text_edit.append(f"\nAI Initialization Error: {e}")
                self.ai_enabled_globally = False
                self.ai_agent = None
        else:
            self.ai_enabled_globally = False
            self.ai_agent = None
        self.update_ai_status_label()

    def save_settings(self):
        """Saves current UI and AI preferences to QSettings."""
        self.settings.setValue('ai_enabled', self.ai_enabled)
        self.settings.setValue('google_api_key', self.google_api_key) # Save the API key
        self.settings.setValue('current_theme', self.current_theme)
        self.settings.setValue('code_font_size', self.code_font_size)
        self.settings.setValue('ai_model_name', self.ai_model_name)
        self.settings.setValue('max_ai_tokens', self.max_ai_tokens)

        self.settings.sync() # Forces changes to be written to permanent storage
        logging.info("Settings saved via QSettings.")

    def update_ai_status_label(self):
        """Updates the AI status label in the status bar."""
        if self.ai_enabled_globally:
            if self.ai_enabled and self.ai_agent and self.ai_agent.api_status == "READY":
                self.ai_status_label.setText("AI Status: <font color='green'>Ready</font>")
                self.ask_ai_general_help_button.setEnabled(True)
                # Enable check_answer_button only if AI is ready and an exercise is active
                if self.current_exercise_index != -1 and self.current_lesson_index != -1:
                    self.check_answer_button.setEnabled(True)
            elif self.ai_enabled and self.ai_agent and self.ai_agent.api_status != "READY":
                self.ai_status_label.setText(f"AI Status: <font color='orange'>Not Ready ({self.ai_agent.api_status})</font>")
                self.ask_ai_general_help_button.setEnabled(False)
                self.check_answer_button.setEnabled(False)
            elif not self.ai_enabled:
                self.ai_status_label.setText("AI Status: <font color='red'>Disabled (User)</font>")
                self.ask_ai_general_help_button.setEnabled(False)
                self.check_answer_button.setEnabled(False)
        else:
            self.ai_status_label.setText("AI Status: <font color='red'>Globally Disabled (No API Key/Error)</font>")
            self.ask_ai_general_help_button.setEnabled(False)
            self.check_answer_button.setEnabled(False)
        # Ensure the toggle action reflects whether it's enabled or not
        self.toggle_ai_action.setEnabled(self.ai_enabled_globally)


    def toggle_ai_features(self):
        """Toggles AI features on/off based on user action."""
        # Only allow toggling if AI is globally enabled (i.e., API key is present and agent initialized)
        if not self.ai_enabled_globally:
            QMessageBox.warning(self, "AI Disabled", "AI features are globally disabled (e.g., no API key or initialization error) and cannot be toggled on by the user.")
            self.toggle_ai_action.setChecked(False) # Ensure checkbox stays unchecked
            return

        self.ai_enabled = self.toggle_ai_action.isChecked()
        self.save_settings() # Save the new AI enabled state
        self.update_ai_status_label() # Update the status label and button states

        if self.ai_enabled:
            self.output_text_edit.append("\nAI features enabled by user.")
            QMessageBox.information(self, "AI Enabled", "AI features are now enabled.")
        else:
            self.output_text_edit.append("\nAI features disabled by user.")
            QMessageBox.information(self, "AI Disabled", "AI features are now disabled.")
        
        # Button states are now managed by update_ai_status_label

    def show_ai_model_settings_dialog(self):
        """Opens a dialog to configure AI model settings."""
        dialog = QDialog(self)
        dialog.setWindowTitle("AI Model Settings")
        layout = QVBoxLayout(dialog)

        # Model Selection
        model_label = QLabel("AI Model:")
        model_combo = QComboBox()
        # Add available Gemini models. 'gemini-pro' is older, 'gemini-1.5-flash-latest' is recommended for speed, 'gemini-1.5-pro-latest' for capability
        model_combo.addItems(['gemini-1.5-flash-latest', 'gemini-1.5-pro-latest', 'gemini-pro'])
        model_combo.setCurrentText(self.ai_model_name)

        model_layout = QHBoxLayout()
        model_layout.addWidget(model_label)
        model_layout.addWidget(model_combo)
        layout.addLayout(model_layout)

        # Max Tokens
        tokens_label = QLabel("Max Output Tokens:")
        tokens_input = QLineEdit(str(self.max_ai_tokens))
        tokens_input.setValidator(QIntValidator(1, 8192)) # Gemini 1.5 models support up to 8192 output tokens

        tokens_layout = QHBoxLayout()
        tokens_layout.addWidget(tokens_label)
        tokens_layout.addWidget(tokens_input)
        layout.addLayout(tokens_layout)

        # API Key Input (allow user to set/update)
        api_key_label = QLabel("Google API Key:")
        api_key_input = QLineEdit(self.google_api_key)
        api_key_input.setPlaceholderText("Enter your Google Gemini API Key")
        api_key_input.setEchoMode(QLineEdit.Password) # Mask the input

        api_key_layout = QHBoxLayout()
        api_key_layout.addWidget(api_key_label)
        api_key_layout.addWidget(api_key_input)
        layout.addLayout(api_key_layout)

        # Buttons
        buttons_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        buttons_layout.addStretch(1)
        buttons_layout.addWidget(ok_button)
        buttons_layout.addWidget(cancel_button)
        layout.addLayout(buttons_layout)

        if dialog.exec_() == QDialog.Accepted:
            new_model = model_combo.currentText()
            new_tokens = int(tokens_input.text())
            new_api_key = api_key_input.text().strip()

            settings_changed = False
            if new_model != self.ai_model_name:
                self.ai_model_name = new_model
                logging.info(f"AI Model updated to: {self.ai_model_name}")
                settings_changed = True

            if new_tokens != self.max_ai_tokens:
                self.max_ai_tokens = new_tokens
                logging.info(f"Max AI Output Tokens updated to: {self.max_ai_tokens}")
                settings_changed = True

            if new_api_key != self.google_api_key:
                self.google_api_key = new_api_key
                logging.info("Google API Key updated.")
                settings_changed = True
                # Re-initialize AI agent immediately if API key changed
                self.initialize_ai_agent(self.google_api_key)
            elif settings_changed and self.ai_agent:
                # If model/tokens changed but not key, update agent directly
                self.ai_agent.update_model(self.ai_model_name)
                self.ai_agent.update_max_tokens(self.max_ai_tokens)

            if settings_changed:
                self.save_settings() # Save changes
                self.update_ai_status_label() # Refresh AI status display

    def show_font_size_dialog(self):
        """Opens a dialog to select font size."""
        current_size = self.code_font_size
        sizes = [8, 9, 10, 11, 12, 14, 16, 18, 20] # Common font sizes
        item, ok = QInputDialog.getItem(self, "Select Font Size", "Font Size:",
                                         [str(s) for s in sizes],
                                         sizes.index(current_size) if current_size in sizes else 2,
                                         False)
        if ok and item:
            new_size = int(item)
            if new_size != current_size:
                self.apply_font_size(new_size)
                self.save_settings() # Save changes immediately
                
    def apply_theme(self, theme_name):
        """Applies the specified theme (Dark or Light) to the application."""
        if theme_name == 'Dark':
            self.apply_dark_theme()
        elif theme_name == 'Light':
            self.apply_light_theme()
        else:
            logging.warning(f"Unknown theme: {theme_name}. Defaulting to Dark.")
            self.apply_dark_theme() # Fallback to dark theme

        self.current_theme = theme_name # Update current_theme after applying

    def apply_dark_theme(self):
        """Applies a dark theme to the application."""
        qss = """
        QMainWindow, QWidget {
            background-color: #2b2b2b; /* Dark grey background */
            color: #f0f0f0; /* Light grey text */
        }
        QTextEdit {
            background-color: #3c3c3c; /* Even darker for code/text areas */
            color: #f0f0f0;
            border: 1px solid #555555;
            border-radius: 5px; /* Rounded corners */
            padding: 5px;
            font-family: "Consolas", "Courier New", monospace;
        }
        QPushButton {
            background-color: #505050;
            color: #ffffff;
            border: 1px solid #666;
            border-radius: 4px;
            padding: 8px 15px;
        }
        QPushButton:hover {
            background-color: #606060;
        }
        QPushButton:pressed {
            background-color: #404040;
        }
        QSplitter::handle {
            background-color: #505050;
        }
        QListWidget {
            background-color: #333;
            color: #f0f0f0;
            border: 1px solid #444;
            border-radius: 5px;
            padding: 5px;
        }
        QListWidget::item:selected {
            background-color: #0078d7; /* Highlight color */
            color: #ffffff;
        }
        QMenuBar {
            background-color: #3c3c3c;
            color: #f0f0f0;
        }
        QMenuBar::item {
            padding: 5px 10px;
            background-color: #3c3c3c;
        }
        QMenuBar::item:selected {
            background-color: #555;
        }
        QMenu {
            background-color: #3c3c3c;
            color: #f0f0f0;
            border: 1px solid #555;
        }
        QMenu::item {
            padding: 5px 20px;
        }
        QMenu::item:selected {
            background-color: #555;
        }
        QLabel {
            color: #f0f0f0;
        }
        QComboBox {
            background-color: #555555;
            color: #E0E0E0;
            border: 1px solid #777777;
            padding: 1px 0px 1px 3px;
            border-radius: 3px;
        }
        QLineEdit {
            background-color: #3c3c3c;
            color: #f0f0f0;
            border: 1px solid #555555;
            border-radius: 3px;
            padding: 2px;
        }
        QStatusBar {
            background-color: #3c3c3c;
            color: #f0f0f0;
            border-top: 1px solid #555;
        }
        /* Specific object names for fonts - ensure they match init_ui object names */
        #lessonContent, #codeEditor, #outputText, #aiQuestionInput {
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 10pt; /* This will be overridden by apply_font_size if different */
        }
        """
        self.setStyleSheet(qss)
        logging.info("Dark theme applied.")

    def apply_light_theme(self):
        """Applies a light theme to the application."""
        qss = """
        QMainWindow, QWidget {
            background-color: #F0F0F0; /* Light grey background */
            color: #333333; /* Dark text */
        }
        QTextEdit {
            background-color: #FFFFFF; /* White for code/text areas */
            color: #000000;
            border: 1px solid #CCCCCC;
            border-radius: 5px; /* Rounded corners */
            padding: 5px;
            font-family: "Consolas", "Courier New", monospace;
        }
        QPushButton {
            background-color: #E0E0E0;
            color: #333333;
            border: 1px solid #B0B0B0;
            border-radius: 4px;
            padding: 8px 15px;
        }
        QPushButton:hover {
            background-color: #D0D0D0;
        }
        QPushButton:pressed {
            background-color: #C0C0C0;
        }
        QSplitter::handle {
            background-color: #CCCCCC;
        }
        QListWidget {
            background-color: #F5F5F5;
            color: #333333;
            border: 1px solid #CCCCCC;
            border-radius: 5px;
            padding: 5px;
        }
        QListWidget::item:selected {
            background-color: #ADD8E6; /* Light blue highlight */
            color: #000000;
        }
        QMenuBar {
            background-color: #E0E0E0;
            color: #333333;
        }
        QMenuBar::item {
            padding: 5px 10px;
            background-color: #E0E0E0;
        }
        QMenuBar::item:selected {
            background-color: #ADD8E6;
        }
        QMenu {
            background-color: #E0E0E0;
            color: #333333;
            border: 1px solid #CCCCCC;
        }
        QMenu::item {
            padding: 5px 20px;
        }
        QMenu::item:selected {
            background-color: #ADD8E6;
        }
        QLabel {
            color: #333333;
        }
        QComboBox {
            background-color: #E0E0E0;
            color: #333333;
            border: 1px solid #B0B0B0;
            padding: 1px 0px 1px 3px;
            border-radius: 3px;
        }
        QLineEdit {
            background-color: #FFFFFF;
            color: #000000;
            border: 1px solid #CCCCCC;
            border-radius: 3px;
            padding: 2px;
        }
        QStatusBar {
            background-color: #E0E0E0;
            color: #333333;
            border-top: 1px solid #B0B0B0;
        }
        /* Specific object names for fonts - ensure they match init_ui object names */
        #lessonContent, #codeEditor, #outputText, #aiQuestionInput {
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 10pt; /* This will be overridden by apply_font_size if different */
        }
        """
        self.setStyleSheet(qss)
        logging.info("Light theme applied.")

    def apply_font_size(self, size):
        """Applies the given font size to the code editor and output text areas."""
        font = QFont()
        font.setPointSize(size)
        
        self.code_editor.setFont(font)
        self.output_text_edit.setFont(font)
        self.lesson_content_text_edit.setFont(font)
        self.ai_question_input.setFont(font) # Apply to AI question input as well

        self.code_font_size = size
        logging.info(f"Font size applied: {size}")

    def show_about_dialog(self):
        """Displays an About dialog with information about the application."""
        QMessageBox.about(
            self,
            "About CodeCompanion V2",
            """
            <h3>CodeCompanion V2</h3>
            <p><strong>Interactive Python Learning Tool with AI Integration</strong></p>
            <p>Version: 2.0.0</p>
            <p>Developed by: [Your Name/Alias Here]</p>
            <p>CodeCompanion V2 is designed to help users learn Python interactively. It features an integrated code editor, lesson display, and AI assistance powered by Google Gemini to provide a rich learning experience.</p>
            <p>For more information, visit the <a href="https://github.com/Snake-eyes82/CodeCompanion-V2">GitHub Repository</a>.</p>
            <p>Built with PySide6 and Google Gemini API.</p>
            """
        )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    tool = PythonLearningTool()
    tool.show()
    sys.exit(app.exec())
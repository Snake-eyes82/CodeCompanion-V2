import sys
import json
import logging
import re
import os
import builtins
import traceback # Added for detailed error logging
import io # Added for capturing output in check_code

# Import from PySide6
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QVBoxLayout, QHBoxLayout,
    QWidget, QPushButton, QLabel, QSplitter, QMessageBox, QListWidget,
    QListWidgetItem, QSizePolicy, QMenuBar
)
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor, QFont, QAction
from PySide6.QtCore import Qt, QSettings

# Import your actual AI agent
from ai_agent import SelfImprovingAgent
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()  # Load environment variables from .env file
class PythonLearningTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Python Learning Tool")
        self.setGeometry(100, 100, 1200, 800) # Initial window size

        # --- INITIALIZE ALL ATTRIBUTES FIRST ---
        self.lessons_data = [] # Stores lesson content from lessons.json
        self.current_lesson_index = -1 # No lesson selected initially
        self.current_exercise_index = 0
        #self.load_user_progress = {} # Stores user scores and attempts
        self.load_user_progress
        self.exercise_attempts = {} # Initialize as an empty dictionary
        # REMOVED: self.load_lesson (no need to store method reference here)
        self.ai_agent = None
        self.ai_enabled_globally = True # Tracks if AI *can* be enabled (API key, init status)
        self.ai_enabled = True # This will be the current toggled state, defaulting to global status

        self.current_exercise_solution_criteria = "" # Stores expected_output for current exercise
        self.current_exercise_check_function = "" # Stores the check_function string for current exercise
        
        # Directory setup
        self.lessons_dir = os.path.join(os.path.dirname(__file__), 'lessons')
        self.lessons_json_path = os.path.join(self.lessons_dir, 'lessons.json')
        self.lessons_generated_dir = os.path.join(self.lessons_dir, 'lessons_generated')
        
        os.makedirs(self.lessons_dir, exist_ok=True)
        os.makedirs(self.lessons_generated_dir, exist_ok=True)

        self.settings = QSettings("PythonLearningTool", "Settings") # For saving user preferences
        # --- END INITIALIZATION ---

        # Load settings here so ai_enabled is set before init_ui/menu
        self.load_settings() # Load settings at startup (including AI toggle)

        # --- CALL init_ui() AND create_menu() AFTER ALL WIDGETS ARE READY ---
        self.init_ui() # This creates self.lesson_list_widget and other UI elements
        self.create_menu() # Create the menu bar (uses self.ai_enabled, self.toggle_ai_action)
        # --- END UI INITIALIZATION ---

        # Initialize AI Agent (after setting up logging and potentially loading settings)
        print(f"DEBUG: GOOGLE_API_KEY from os.environ: {os.environ.get('GOOGLE_API_KEY')}")
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logging.warning("GOOGLE_API_KEY environment variable not set. AI features will be disabled.")
            self.ai_enabled_globally = False

        if self.ai_enabled_globally:
            try:
                # Assuming SelfImprovingAgent is defined and imported elsewhere
                self.ai_agent = SelfImprovingAgent(api_key=api_key)
                if self.ai_agent.api_status != "READY":
                    logging.error(f"AI Agent initialization failed: {self.ai_agent.api_status}. AI features disabled.")
                    QMessageBox.critical(self, "AI Initialization Failed", f"AI Agent could not be initialized: {self.ai_agent.api_status}. AI features will be disabled. Check your internet connection or API key validity.")
                    self.ai_enabled_globally = False
                else:
                    logging.info("AI Agent initialized successfully.")
            except Exception as e:
                logging.error(f"Unexpected error during AI Agent initialization: {e}", exc_info=True)
                QMessageBox.critical(self, "AI Initialization Error", f"An unexpected error occurred during AI Agent initialization: {e}. AI features will be disabled.")
                self.ai_enabled_globally = False
        
        # Conditional lesson loading/generation
        if not os.path.exists(self.lessons_json_path):
            logging.info("lessons.json not found. Attempting to generate initial lessons.")
            if self.ai_enabled_globally and self.ai_agent and self.ai_agent.api_status == "READY":
                self.generate_initial_lessons_with_ai()
            else:
                logging.warning("AI is not available to generate initial lessons. Application might be limited.")
                QMessageBox.warning(self, "No Lessons Found", "No lessons found and AI is not available to generate them. Application functionality will be limited.")

        # This will now call the unified load_lesson method:
        # It should (1) load all lessons from lessons.json into self.lessons_data
        # AND (2) populate the self.lesson_list_widget
        # AND (3) if lessons exist, display the first one (index 0).
        self.load_lesson() # Call without arguments for initial load of all lessons & display of first.
        self.load_user_progress() # Load progress after lessons are available

        self.update_ai_status_label() # Update AI status label after all initialization
        
    def load_user_progress(self):
        """Loads user progress from a JSON file."""
        progress_file = "user_progress.json"
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    self.user_progress = json.load(f)
                logging.info("User progress loaded successfully.")
                self.output_text_edit.append("User progress loaded.")
                
                # Restore exercise attempts from loaded progress
                self.exercise_attempts = self.user_progress.get('exercise_attempts', {})

                # You might also want to restore current lesson/exercise from here
                # Example:
                # self.current_lesson_index = self.user_progress.get('current_lesson_index', 0)
                # self.current_exercise_index = self.user_progress.get('current_exercise_index', 0)
                # self.load_lesson(self.current_lesson_index) # Reload the last active lesson/exercise
                # self.display_exercise(self.current_exercise_index)
            except Exception as e:
                logging.error(f"Error loading user progress: {e}", exc_info=True)
                self.output_text_edit.append(f"<font color='red'>Error loading user progress: {e}</font>")
                self.user_progress = {} # Reset to empty if corrupted
                self.exercise_attempts = {}
        else:
            logging.info("user_progress.json not found. Starting with fresh progress.")
            self.user_progress = {}
            self.exercise_attempts = {}

    def save_user_progress(self):
        """Saves current user progress to a JSON file."""
        progress_file = "user_progress.json"
        try:
            # Prepare data to save
            data_to_save = {
                'exercise_attempts': self.exercise_attempts,
                # You might add other progress data here, e.g.:
                # 'completed_exercises': self.completed_exercises,
                # 'current_lesson_index': self.current_lesson_index,
                # 'current_exercise_index': self.current_exercise_index,
            }
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=4)
            logging.info("User progress saved successfully.")
        except Exception as e:
            logging.error(f"Error saving user progress: {e}", exc_info=True)
            self.output_text_edit.append(f"<font color='red'>Error saving user progress: {e}</font>")

    def closeEvent(self, event):
        """Overrides the close event to save user progress."""
        self.save_user_progress()
        event.accept() # Accept the close event, allowing the window to close

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Left Panel (Lessons List)
        self.left_panel_layout = QVBoxLayout()
        self.lessons_label = QLabel("Lessons:")
        self.lessons_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.left_panel_layout.addWidget(self.lessons_label)

        # Lessons List Widget
        self.lesson_list_widget = QListWidget()
        self.lesson_list_widget.setObjectName("lessonList") # For styling
        self.lesson_list_widget.setMinimumWidth(180)
        self.lesson_list_widget.setMaximumWidth(250)
        # CORRECTED CONNECTION: Use itemClicked and a handler (lesson_list_item_clicked)
        # This handler will then call load_lesson with the appropriate index.
        self.lesson_list_widget.itemClicked.connect(self.lesson_list_item_clicked)
        self.left_panel_layout.addWidget(self.lesson_list_widget)

        self.left_widget = QWidget()
        self.left_widget.setLayout(self.left_panel_layout)
        self.left_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # Right Panel (Lesson Content, Code Editor, Output, AI Question)
        self.right_panel_layout = QVBoxLayout()

        # Lesson Content Text Edit
        self.lesson_content_text_edit = QTextEdit()
        self.lesson_content_text_edit.setReadOnly(True)
        self.lesson_content_text_edit.setPlaceholderText("Select a lesson to view its content.")
        self.lesson_content_text_edit.setObjectName("lessonContent") # Assign object name for CSS

        # Code Editor
        self.code_editor = QTextEdit()
        self.code_editor.setFont(QFont("Console", 10))
        # Ensure apply_syntax_highlighting is defined later in your class
        self.code_editor.textChanged.connect(self.apply_syntax_highlighting)
        self.code_editor.setPlaceholderText("Write your Python code here...")
        self.code_editor.setObjectName("codeEditor") # Assign object name for CSS

        # Run and Check Buttons
        self.button_layout = QHBoxLayout()
        self.run_code_button = QPushButton("Run Code") # Make sure this is present!
        # Ensure run_user_code is defined later in your class
        self.run_code_button.clicked.connect(self.run_user_code)
        self.button_layout.addWidget(self.run_code_button)

        self.check_answer_button = QPushButton("Check Answer") # Make sure this is present!
        # Connect to the correct check_answer method
        # self.check_answer_button.clicked.connect(lambda: self.check_answer_with_ai(self.code_editor.toPlainText())) # Old connection
        self.check_answer_button.clicked.connect(lambda: self.check_answer_with_ai(self.code_editor.toPlainText())) # Use the unified check_answer
        self.button_layout.addWidget(self.check_answer_button)
        self.right_panel_layout.addLayout(self.button_layout) # Add the button_layout to the right_panel_layout

        # Output Text Edit
        self.output_text_edit = QTextEdit()
        self.output_text_edit.setReadOnly(True)
        self.output_text_edit.setPlaceholderText("Code output and AI feedback will appear here.")
        self.output_text_edit.setObjectName("outputText") # Assign object name for CSS

        # Ask AI section (matches image layout)
        self.ask_ai_label = QLabel("Ask The AI:") # Make sure this is present!
        self.ask_ai_label.setFont(QFont("Segoe UI", 10, QFont.Bold))

        self.ai_question_input = QTextEdit() # Make sure this is present!
        self.ai_question_input.setPlaceholderText("Type your question here (e.g., 'What is a variable?', 'How does def work?')")
        self.ai_question_input.setObjectName("aiQuestionInput") # Assign object name for CSS
        self.ai_question_input.setMaximumHeight(80)
        
        self.ask_ai_general_help_button = QPushButton("Ask AI for General Help") # Make sure this is present!
        # Ensure ask_ai_general_help is defined later in your class
        self.ask_ai_general_help_button.clicked.connect(self.ask_ai_general_help)

        # Nested Splitter for Right Panel
        self.right_splitter = QSplitter(Qt.Vertical)
        self.right_splitter.addWidget(self.lesson_content_text_edit)
        self.right_splitter.addWidget(self.code_editor)
        self.right_splitter.addWidget(self.output_text_edit)
        
        # Wrap AI question input and button in a widget for splitter
        ai_question_area_widget = QWidget()
        ai_question_area_layout = QVBoxLayout(ai_question_area_widget)
        ai_question_area_layout.setContentsMargins(0,0,0,0) # Remove extra margins
        ai_question_area_layout.addWidget(self.ask_ai_label)
        ai_question_area_layout.addWidget(self.ai_question_input)
        ai_question_area_layout.addWidget(self.ask_ai_general_help_button)
        self.right_splitter.addWidget(ai_question_area_widget)

        self.right_splitter.setSizes([250, 250, 150, 100]) # Example distribution for content, code, output, AI
        self.right_panel_layout.addWidget(self.right_splitter) # Add the splitter to the main right layout


        # --- VITAL: ADD NAVIGATION BUTTONS HERE ---
        # Lesson and Exercise Navigation Buttons
        self.nav_button_layout = QHBoxLayout() # Create a new horizontal layout for navigation buttons

        self.prev_exercise_button = QPushButton("Previous Exercise") # <--- ADD THIS
        self.prev_exercise_button.clicked.connect(self.prev_exercise)
        self.nav_button_layout.addWidget(self.prev_exercise_button) # <--- ADD THIS

        self.next_exercise_button = QPushButton("Next Exercise") # <--- ADD THIS
        self.next_exercise_button.clicked.connect(self.next_exercise)
        self.nav_button_layout.addWidget(self.next_exercise_button) # <--- ADD THIS

        self.prev_lesson_button = QPushButton("Previous Lesson") # <--- ADD THIS
        self.prev_lesson_button.clicked.connect(self.prev_lesson)
        self.nav_button_layout.addWidget(self.prev_lesson_button) # <--- ADD THIS

        self.next_lesson_button = QPushButton("Next Lesson") # <--- ADD THIS
        self.next_lesson_button.clicked.connect(self.next_lesson)
        self.nav_button_layout.addWidget(self.next_lesson_button) # <--- ADD THIS

        self.right_panel_layout.addLayout(self.nav_button_layout) # <--- ADD THIS to add the nav layout to the right panel
        # --- END VITAL NAVIGATION BUTTONS SECTION ---


        self.right_widget = QWidget()
        self.right_widget.setLayout(self.right_panel_layout)
        self.right_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Main Splitter to divide left (lessons list) and right (content, code, output, ai) panels
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.addWidget(self.left_widget)
        self.main_splitter.addWidget(self.right_widget)
        self.main_splitter.setSizes([200, 1000]) # Initial distribution for left list and right content
        self.main_layout.addWidget(self.main_splitter)

        # Status Bar for score (ensure this is present too!)
        self.status_bar = self.statusBar()
        self.score_label = QLabel("Score: N/A") # Make sure this is present!
        self.status_bar.addPermanentWidget(self.score_label)

        # Apply the styles for the window boxes and dark theme
        # Ensure apply_window_box_styles is defined later in your class
        self.apply_window_box_styles()
        
        # --- VITAL: AI Status Label - ENSURE THIS IS PRESENT AND UNCOMMENTED! ---
        self.ai_status_label = QLabel("AI Status: Initializing...") # <--- ADD THIS
        self.statusBar().addPermanentWidget(self.ai_status_label) # <--- ADD THIS
        # --- END VITAL AI STATUS LABEL SECTION ---

        # Initial status update handled after settings load in __init__
        # These state updates MUST be called AFTER all relevant buttons are created in init_ui
        self.update_exercise_buttons_state() # Will now correctly find buttons
        self.update_navigation_buttons()     # Will now correctly find buttons
        

    def create_menu(self):
        """Creates the application's menu bar."""
        menu_bar = self.menuBar()

        # File Menu
        file_menu = menu_bar.addMenu("&File")
        exit_action = QAction("&Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Settings Menu
        settings_menu = menu_bar.addMenu("&Settings")
        self.toggle_ai_action = QAction("Enable AI", self, checkable=True)
        self.toggle_ai_action.setChecked(self.ai_enabled)
        self.toggle_ai_action.triggered.connect(self.toggle_ai_features)
        settings_menu.addAction(self.toggle_ai_action)

        # Help Menu
        help_menu = menu_bar.addMenu("&Help")
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

    def toggle_ai_features(self, checked):
        """Toggles AI features on/off and updates status."""
        self.ai_enabled = checked
        self.save_settings() # Save the AI enabled state
        self.update_ai_status_label() # Update UI elements based on new state
        self.output_text_edit.append(f"\nAI features are now {'Enabled' if self.ai_enabled else 'Disabled'}.")
        logging.info(f"AI features toggled to: {self.ai_enabled}")


    def generate_initial_lessons_with_ai(self):
        """Generates initial lessons using the AI agent if lessons.json is not found."""
        if not (self.ai_agent and self.ai_agent.api_status == "READY"): # Check ai_agent directly
            logging.warning("AI Agent not ready for lesson generation.")
            self.output_text_edit.append("\nAI Agent not ready. Cannot generate initial lessons.")
            QMessageBox.warning(self, "AI Not Ready", "AI features are not ready to generate lessons. Ensure API key is set and internet connection is stable.")
            return False

        try:
            logging.info("AI: Attempting to generate initial lesson content.")
            initial_topics = [
                {"title": "Introduction to Python: Hello World!", "id": "intro_to_python_hello_world"},
                {"title": "Variables and Data Types", "id": "variables_data_types"},
                {"title": "Control Structures: If Statements and Loops", "id": "control_structures"},
                {"title": "Functions and Modules", "id": "functions_and_modules"},
                {"title": "Data Structures: Lists, Tuples, and Dictionaries", "id": "data_structures"},
                {"title": "File Handling in Python", "id": "file_handling"},
                {"title": "Error Handling and Exceptions", "id": "error_handling"},
                {"title": "Object-Oriented Programming Basics", "id": "oop_basics"}
            ]

            generated_lessons_data = []

            for i, topic in enumerate(initial_topics):
                self.output_text_edit.append(f"\nAI: Generating lesson for '{topic['title']}'...")
                if QApplication.instance():
                    QApplication.instance().processEvents() 

                ai_response_dict = self.ai_agent.generate_lesson_content(topic["title"])

                # --- Input Validation for AI Response ---
                if not isinstance(ai_response_dict, dict):
                    logging.error(f"AI generation for '{topic['title']}' returned non-dict response: {ai_response_dict}")
                    self.output_text_edit.append(f"AI Error: Invalid response format for '{topic['title']}'. Skipping.")
                    continue
                
                if "error_message" in ai_response_dict:
                    err_msg = ai_response_dict.get("error_message", "Unknown error")
                    logging.error(f"AI generation failed for '{topic['title']}': {err_msg}")
                    self.output_text_edit.append(f"AI Error: Failed to generate '{topic['title']}': {err_msg}. Skipping this lesson.")
                    continue

                if "lesson_content_markdown" not in ai_response_dict or "exercises" not in ai_response_dict:
                    logging.error(f"AI response for '{topic['title']}' missing 'lesson_content_markdown' or 'exercises' keys. Response: {ai_response_dict}")
                    self.output_text_edit.append(f"AI Error: Incomplete response for '{topic['title']}'. Skipping.")
                    continue

                lesson_content = ai_response_dict["lesson_content_markdown"]
                exercises = ai_response_dict["exercises"]
                
                if not isinstance(lesson_content, str):
                    logging.warning(f"AI returned non-string lesson_content_markdown for '{topic['title']}'. Converting to string.")
                    lesson_content = str(lesson_content)

                if not isinstance(exercises, list):
                    logging.warning(f"AI returned non-list exercises for '{topic['title']}'. Skipping exercises for this lesson.")
                    exercises = [] # Ensure it's an empty list if not a list
                
                logging.debug(f"DEBUG: After extracting exercises: Type={type(exercises)}, Length={len(exercises) if isinstance(exercises, list) else 'Not a list'}")
                if isinstance(exercises, list):
                    logging.debug(f"DEBUG: First few exercises (if any): {exercises[:2]}")

                md_filename = f"{topic['id']}.md"
                md_filepath = os.path.join(self.lessons_generated_dir, md_filename)
                try:
                    with open(md_filepath, 'w', encoding='utf-8') as f:
                        f.write(lesson_content)
                    logging.info(f"AI: Lesson content saved to {md_filepath}")
                except IOError as e:
                    logging.error(f"Could not write lesson file {md_filepath}: {e}")
                    self.output_text_edit.append(f"Error: Could not save lesson file for '{topic['title']}'. Skipping.")
                    continue # Skip to next topic if file cannot be saved

                exercise_checks_for_json = []
                logging.debug(f"DEBUG: Before exercise loop: exercise_checks_for_json length={len(exercise_checks_for_json)}")
                
                # Use 'exercises' directly as it's already type-checked to be a list
                for j, exercise in enumerate(exercises):
                    if not isinstance(exercise, dict):
                        logging.warning(f"Skipping non-dict exercise item {j} in '{topic['title']}': {exercise}")
                        continue # Skip to next exercise if it's not a dictionary

                    logging.debug(f"DEBUG: Inside exercise loop, processing exercise {j}: Keys={exercise.keys()}")
                    
                    # Use str() conversion with .strip() for robustness
                    check_func_str = str(exercise.get('check_function', '')).strip()
                    initial_code_str = str(exercise.get('initial_code', '')).strip()
                    prompt_str = str(exercise.get('prompt', 'No prompt provided.')).strip() # Explicitly convert to str
                    
                    if not check_func_str or "def check_result(" not in check_func_str:
                        logging.warning(f"AI generated an invalid check_function for '{topic['title']}' exercise {j}. Using fallback checker.")
                        check_func_str = (
                            "def check_result(user_code, expected_output):\n"
                            "    import io, sys\n"
                            "    old_stdout = sys.stdout\n"
                            "    redirected_output = io.StringIO()\n"
                            "    sys.stdout = redirected_output\n"
                            "    try:\n"
                            "        exec_globals = {'__builtins__': __builtins__}\n"
                            "        exec(user_code, exec_globals)\n"
                            "        output = redirected_output.getvalue().strip()\n"
                            "        \n"
                            "        if output == expected_output.strip():\n"
                            "            return {'passed': True, 'score': 1.0, 'message': 'Code passed basic output test.'}\n"
                            "        else:\n"
                            "            return {'passed': False, 'score': 0.5, 'message': f'Output mismatch. Expected: \"{expected_output.strip()}\", Got: \"{output}\".'}\n"
                            "    except Exception as e:\n"
                            "        return {'passed': False, 'score': 0.0, 'message': f'Code execution error: {e}'}\n"
                            "    finally:\n"
                            "        sys.stdout = old_stdout\n"
                        )
                    
                    expected_output_val = exercise.get('expected_output')
                    # This logic is already good, just ensuring it's kept as is.
                    if expected_output_val is None:
                        expected_output_string = ""
                    else:
                        expected_output_string = str(expected_output_val).strip()


                    exercise_checks_for_json.append({
                        "prompt": prompt_str, # Use the explicitly converted string
                        "initial_code": initial_code_str,
                        "check_function": check_func_str,
                        "expected_output": expected_output_string
                    })
                logging.debug(f"DEBUG: After exercise loop: exercise_checks_for_json length={len(exercise_checks_for_json)}")

                if not exercise_checks_for_json:
                    logging.warning(f"No valid exercises processed for lesson '{topic['title']}'. Skipping this lesson entry.")
                    self.output_text_edit.append(f"AI Warning: No valid exercises for '{topic['title']}'. Skipping lesson.")
                    continue # Skip this lesson if no exercises were successfully processed

                lesson_entry = {
                    "id": topic["id"],
                    "title": topic["title"],
                    "content_file": os.path.join("lessons", "lessons_generated", md_filename).replace(os.sep, '/'),
                    "exercises": exercise_checks_for_json,
                    "solution_criteria": exercise_checks_for_json[0]['expected_output'] if exercise_checks_for_json else "No specific solution criteria defined."
                }
                generated_lessons_data.append(lesson_entry)
                self.output_text_edit.append(f"AI: Successfully processed lesson '{topic['title']}'.")

            # Save the master lessons.json only if some lessons were successfully generated
            if generated_lessons_data:
                with open(self.lessons_json_path, 'w', encoding='utf-8') as f:
                    json.dump(generated_lessons_data, f, indent=4)
                logging.info(f"AI: Generated lessons.json saved to {self.lessons_json_path}")
                self.output_text_edit.append("\nAI: Initial lessons generation complete.")
                self.load_lesson() # Refresh UI to show new lessons
                return True
            else:
                logging.warning("No lessons were successfully generated by AI. lessons.json not created/updated.")
                self.output_text_edit.append("\nAI: No lessons could be generated. Check AI response format and API status.")
                QMessageBox.warning(self, "No Lessons Generated", "AI failed to generate any valid lessons. Check logs and API key.")
                return False

        except json.JSONDecodeError as e:
            logging.error(f"AI: Failed to parse AI-generated JSON response for initial lessons: {e}", exc_info=True)
            self.output_text_edit.append(f"<font color='red'>Error: Failed to parse AI-generated initial lesson data. {e}</font>")
            QMessageBox.critical(self, "Lesson Generation Error", f"Failed to parse AI response: {e}")
            return False
        except Exception as e:
            logging.error(f"AI: Unexpected error during initial lesson generation process: {e}", exc_info=True)
            self.output_text_edit.append(f"<font color='red'>An unexpected error occurred during AI lesson generation: {e}</font>")
            QMessageBox.critical(self, "Lesson Generation Error", f"An unexpected error occurred: {e}")
            return False

    def load_lesson(self, lesson_index=None):
        """
        If lesson_index is None: Loads all lessons from lessons.json into self.lessons_data,
        populates the lesson list widget, and then displays the first lesson.
        If lesson_index is provided: Displays the content of the specified lesson
        from self.lessons_data.
        """
        if not self.lessons_data or lesson_index is None:
            self.lesson_list_widget.clear()

            lessons_file = self.lessons_json_path
            try:
                with open(lessons_file, 'r', encoding='utf-8') as f:
                    self.lessons_data = json.load(f)
                logging.info(f"Loaded {len(self.lessons_data)} lessons from {lessons_file}.")
                self.output_text_edit.append(f"Loaded {len(self.lessons_data)} lessons.")

                for idx, lesson in enumerate(self.lessons_data):
                    item = QListWidgetItem(lesson.get('title', f"Lesson {idx + 1}"))
                    self.lesson_list_widget.addItem(item)

            except FileNotFoundError:
                logging.warning(f"Lessons file not found at {lessons_file}. This might be expected if generating.")
                self.lessons_data = []
                self.output_text_edit.append("<font color='orange'>Warning: 'lessons.json' not found. Please ensure it exists or generate lessons.</font>")
                self.lesson_content_text_edit.setMarkdown("<h2>No Lessons Available</h2><p>Please generate lessons or ensure 'lessons.json' exists.</p>")
                self.code_editor.setPlainText("")
                self.output_text_edit.append("")
                self.current_lesson_index = -1
                self.current_exercise_index = -1
                self.update_navigation_buttons()
                self.update_exercise_buttons_state()
                return
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding lessons.json: {e}")
                self.lessons_data = []
                self.output_text_edit.append(f"<font color='red'>Error: Could not read lessons.json. Invalid format: {e}</font>")
                self.lesson_content_text_edit.setMarkdown("<h2>Error Loading Lessons</h2><p>Invalid JSON format. Check 'lessons.json'.</p>")
                self.code_editor.setPlainText("")
                self.output_text_edit.append("")
                self.current_lesson_index = -1
                self.current_exercise_index = -1
                self.update_navigation_buttons()
                self.update_exercise_buttons_state()
                return
            except Exception as e:
                logging.error(f"An unexpected error occurred while loading lessons: {e}")
                self.lessons_data = []
                self.output_text_edit.append(f"<font color='red'>Error: An unexpected error occurred while loading lessons: {e}</font>")
                self.lesson_content_text_edit.setMarkdown("<h2>Error Loading Lessons</h2><p>An unexpected error occurred.</p>")
                self.code_editor.setPlainText("")
                self.output_text_edit.append("")
                self.current_lesson_index = -1
                self.current_exercise_index = -1
                self.update_navigation_buttons()
                self.update_exercise_buttons_state()
                return
            
            if lesson_index is None and self.lessons_data:
                lesson_index = 0

        if not self.lessons_data or not (0 <= lesson_index < len(self.lessons_data)):
            logging.debug(f"No lessons data or invalid index ({lesson_index}). Not attempting to display lesson content.")
            self.lesson_content_text_edit.setMarkdown("<h2>No Lesson Selected</h2><p>Select a lesson from the list, or generate new ones.</p>")
            self.code_editor.setPlainText("")
            # self.exercise_prompt_label.setText("") # Assuming you have this label
            self.current_lesson_index = -1
            self.current_exercise_index = -1
            self.update_navigation_buttons()
            self.update_exercise_buttons_state()
            return

        self.current_lesson_index = lesson_index
        lesson = self.lessons_data[self.current_lesson_index]

        self.lesson_content_text_edit.clear()
        self.lesson_content_text_edit.setMarkdown(f"<h1>{lesson.get('title', 'Untitled Lesson')}</h1>\n"
                                                 f"<h3>{lesson.get('main_concept', 'No main concept provided.')}</h3>\n")
        
        content_file_path = os.path.join(os.path.dirname(__file__), lesson['content_file'])
        try:
            with open(content_file_path, "r", encoding="utf-8") as f:
                md_content = f.read()
            self.lesson_content_text_edit.append(md_content)
        except FileNotFoundError:
            logging.error(f"Lesson content markdown file not found: {content_file_path}")
            self.lesson_content_text_edit.append(f"<p><font color='red'>Error: Lesson content file not found at `{content_file_path}`.</font></p>")
        except Exception as e:
            logging.error(f"Error reading lesson content markdown: {e}")
            self.lesson_content_text_edit.append(f"<p><font color='red'>Error reading lesson content: {e}</font></p>")

        self.setWindowTitle(f"Python Learning Tool - {lesson.get('title', 'Untitled Lesson')}")
        self.output_text_edit.clear()

        if self.lesson_list_widget.currentRow() != self.current_lesson_index:
            self.lesson_list_widget.setCurrentRow(self.current_lesson_index)

        exercises = lesson.get('exercises', [])
        if exercises:
            self.display_exercise(0)
        else:
            self.output_text_edit.append("\nNo exercises for this lesson.")
            # self.exercise_prompt_label.setText("No exercises for this lesson.") # Assuming you have this label
            self.code_editor.setPlainText("")
            self.current_exercise_index = -1
            
        self.update_navigation_buttons()
        self.update_exercise_buttons_state()

    def lesson_list_item_clicked(self, item):
        index = self.lesson_list_widget.row(item)
        self.load_lesson(index)

    def display_exercise(self, exercise_index):
        """Displays the content of a specific exercise within the current lesson."""
        if self.current_lesson_index == -1 or not self.lessons_data:
            # self.exercise_prompt_label.setText("No lesson selected to display exercises.") # Assuming this label exists
            self.code_editor.setPlainText("")
            return

        lesson = self.lessons_data[self.current_lesson_index]
        exercises = lesson.get('exercises', [])

        if not exercises or not (0 <= exercise_index < len(exercises)):
            logging.warning(f"Cannot display exercise: Invalid exercise index {exercise_index} for lesson {self.current_lesson_index}.")
            # self.exercise_prompt_label.setText("No valid exercise to display for this lesson.")
            self.code_editor.setPlainText("")
            self.current_exercise_index = -1
            return

        self.current_exercise_index = exercise_index
        exercise = exercises[self.current_exercise_index]

        # Ensure you have a QLabel named self.exercise_prompt_label in your init_ui
        # If not, you might need to append to output_text_edit or use a different widget.
        # self.exercise_prompt_label.setText(exercise.get('prompt', 'No exercise prompt provided.')) 
        self.code_editor.setPlainText(exercise.get('initial_code', '# Write your code here.'))
        self.current_exercise_solution_criteria = exercise.get('expected_output', '')
        self.current_exercise_check_function = exercise.get('check_function', '')

        self.output_text_edit.append(f"\n--- Exercise {self.current_exercise_index + 1} of {len(exercises)} ---\n")
        self.output_text_edit.append(exercise.get('prompt', ''))
        self.output_text_edit.append("\n-----------------------------------\n")

        self.update_exercise_buttons_state()

    def check_answer_with_ai(self, user_code: str):
        """
        Executes the user's code, captures its output and state, and then
        passes these results to the AI-generated check function for evaluation.
        """
        if not self.ai_enabled or not self.ai_agent or self.ai_agent.api_status != "READY":
            self.output_text_edit.append("\nAI features are currently disabled. Cannot check answer with AI.")
            return

        if not self.lessons_data or self.current_lesson_index == -1:
            self.output_text_edit.append("\nNo lesson selected to check the answer against.")
            return

        current_lesson_data = self.lessons_data[self.current_lesson_index]
        exercises = current_lesson_data.get('exercises', [])

        if not exercises:
            self.output_text_edit.append("\nNo exercises defined for this lesson to check.")
            return

        # CRITICAL FIX: Use self.current_exercise_index to select the correct exercise
        if not (0 <= self.current_exercise_index < len(exercises)):
            self.output_text_edit.append(f"\nError: No current exercise selected or invalid index {self.current_exercise_index}.")
            return
            
        current_exercise = exercises[self.current_exercise_index]
        check_function_str = current_exercise.get('check_function', '').strip()
        expected_output = current_exercise.get('expected_output', '').strip()
        problem_description = current_exercise.get('prompt', 'No problem description provided.')

        if not check_function_str:
            self.output_text_edit.append("\nError: No automated check function available for this exercise.")
            self.output_text_edit.append("\nRequesting AI feedback instead...")
            self.get_ai_feedback_on_code(user_code, problem_description, expected_output)
            return

        self.output_text_edit.append("\n--- Checking Answer with Automated Function ---")
        self.output_text_edit.append("Running your code through the automated checker...\n")
        
        # --- PHASE 1: Execute User's Code in a Controlled Environment ---
        user_output_buffer = io.StringIO()
        user_exec_globals = {'__builtins__': builtins} 
        user_exec_locals = {} 

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = user_output_buffer
        sys.stderr = user_output_buffer 

        user_code_execution_error = None
        try:
            exec(user_code, user_exec_globals, user_exec_locals)
        except Exception as e:
            user_code_execution_error = e
            self.output_text_edit.append(f"User Code Execution Error: {e}\n")
            self.output_text_edit.append(user_output_buffer.getvalue().strip())
            self.output_text_edit.append(f"\nTraceback:\n{traceback.format_exc()}")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        captured_user_output = user_output_buffer.getvalue().strip()
        
        # --- PHASE 2: Execute AI-Generated Check Function ---
        result = {'passed': False, 'score': 0.0, 'message': 'User code failed to execute, cannot check.'}
        if user_code_execution_error:
            result['message'] = f"User code execution failed: {user_code_execution_error}. Output: '{captured_user_output}'"
        else:
            checker_globals = {
                '__builtins__': builtins, 
                'io': io,
                'sys': sys,
                'math': __import__('math'),
                'random': __import__('random'),
                'traceback': traceback,
            }
            
            check_output_buffer = io.StringIO()
            sys.stdout = check_output_buffer
            sys.stderr = check_output_buffer

            try:
                exec(check_function_str, checker_globals) 

                check_result_func = checker_globals.get('check_result')

                if not check_result_func or not callable(check_result_func):
                    raise ValueError("AI-generated check function did not define a callable 'check_result'.")
                
                result = check_result_func(user_code, expected_output)

                if not isinstance(result, dict) or 'passed' not in result or 'score' not in result or 'message' not in result:
                    raise ValueError(f"Automated check function returned invalid format. Got: {result}")
                
            except Exception as e:
                error_output = check_output_buffer.getvalue()
                logging.error(f"Error during AI-generated check function execution: {e}\n{traceback.format_exc()}", exc_info=True)
                result = {'passed': False, 'score': 0.0, 'message': f"An error occurred while running the automated checker: {e}. Checker output: {error_output.strip()}"}
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                if check_output_buffer.getvalue().strip():
                    self.output_text_edit.append(f"\nChecker's internal output:\n{check_output_buffer.getvalue().strip()}")

        # --- PHASE 3: Display Results and Request AI Feedback ---
        self.update_score_label(f"Score: {result['score']:.2f}")
        self.output_text_edit.append(f"\nAI Check Result: {result['message']}")
        
        if result['passed']:
            self.output_text_edit.append("\nCorrect! Moving to next exercise.")
            self.current_exercise_index += 1
            lesson_id = current_lesson_data.get('id', f"lesson_{self.current_lesson_index}")
            completed_exercise_key = f"{lesson_id}-{self.current_exercise_index - 1}"
            self.exercise_attempts[completed_exercise_key] = 0 

            self.ai_agent.learn_from_experience(True, user_code, completed_exercise_key, "")

            if self.current_exercise_index < len(exercises):
                self.display_exercise(self.current_exercise_index)
            else:
                self.output_text_edit.append("Congratulations! You've completed all exercises for this lesson.")
                self.code_editor.setPlainText("")
                # self.exercise_prompt_label.setText("Lesson completed!")
                self.update_navigation_buttons()
                self.update_exercise_buttons_state()
        else:
            self.output_text_edit.append("\nIncorrect. Please try again.")
            lesson_id = current_lesson_data.get('id', f"lesson_{self.current_lesson_index}")
            exercise_id = self.current_exercise_index
            full_exercise_key = f"{lesson_id}-{exercise_id}"
            self.exercise_attempts[full_exercise_key] = self.exercise_attempts.get(full_exercise_key, 0) + 1
            num_attempts = self.exercise_attempts[full_exercise_key]

            feedback_message = self.ai_agent.provide_feedback_on_code(
                user_code=user_code,
                problem_description=problem_description,
                expected_output=expected_output,
                previous_errors=result['message'],
                num_attempts=num_attempts
            )
            self.output_text_edit.append(f"\n--- Agent Feedback ---\n{feedback_message}")
            self.ai_agent.learn_from_experience(False, user_code, full_exercise_key, feedback_message)

        self.update_exercise_buttons_state()
        self.update_navigation_buttons()


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
                num_attempts=1
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
        if self.current_lesson_index == -1 or not self.lessons_data:
            # self.exercise_prompt_label.setText("No lesson selected.")
            self.code_editor.setPlainText("")
            return

        lesson = self.lessons_data[self.current_lesson_index]
        exercises = lesson.get('exercises', [])

        if not exercises:
            # self.exercise_prompt_label.setText("No exercises for this lesson.")
            self.code_editor.setPlainText("")
            return

        if 0 <= self.current_exercise_index < len(exercises):
            self.display_exercise(self.current_exercise_index)
        else:
            self.output_text_edit.append("All exercises for this lesson completed!")
            # self.exercise_prompt_label.setText("All exercises for this lesson completed!")
            self.code_editor.setPlainText("# Lesson Completed!")
            if hasattr(self, 'check_answer_button'): # Check if button exists before disabling
                self.check_answer_button.setEnabled(False)

        self.update_exercise_buttons_state()

    # --- REMAINING PLACEHOLDER METHODS (ensure these are present in your class) ---

    def update_navigation_buttons(self):
        # Logic to enable/disable prev/next lesson buttons
        if hasattr(self, 'prev_lesson_button') and hasattr(self, 'next_lesson_button'): # Check if buttons exist
            if not self.lessons_data:
                self.prev_lesson_button.setEnabled(False)
                self.next_lesson_button.setEnabled(False)
            else:
                self.prev_lesson_button.setEnabled(self.current_lesson_index > 0)
                self.next_lesson_button.setEnabled(self.current_lesson_index < len(self.lessons_data) - 1)
        else:
            logging.debug("Lesson navigation buttons not initialized.")

    def update_exercise_buttons_state(self):
        # Logic to enable/disable prev/next exercise buttons and check answer button
        if hasattr(self, 'prev_exercise_button') and hasattr(self, 'next_exercise_button') and hasattr(self, 'check_answer_button'):
            if self.current_lesson_index == -1 or not self.lessons_data:
                self.prev_exercise_button.setEnabled(False)
                self.next_exercise_button.setEnabled(False)
                self.check_answer_button.setEnabled(False)
                return

            current_lesson = self.lessons_data[self.current_lesson_index]
            exercises = current_lesson.get('exercises', [])

            if not exercises or self.current_exercise_index == -1:
                self.prev_exercise_button.setEnabled(False)
                self.next_exercise_button.setEnabled(False)
                self.check_answer_button.setEnabled(False)
            else:
                self.prev_exercise_button.setEnabled(self.current_exercise_index > 0)
                self.next_exercise_button.setEnabled(self.current_exercise_index < len(exercises) - 1)
                self.check_answer_button.setEnabled(True) # Enable check button if there's an active exercise
        else:
            logging.debug("Exercise navigation or check buttons not initialized.")

    def next_exercise(self):
        if self.current_lesson_index != -1 and self.lessons_data:
            lesson = self.lessons_data[self.current_lesson_index]
            exercises = lesson.get('exercises', [])
            if self.current_exercise_index < len(exercises) - 1:
                self.display_exercise(self.current_exercise_index + 1)
            else:
                self.output_text_edit.append("Already on the last exercise of this lesson.")

    def prev_exercise(self):
        if self.current_lesson_index != -1 and self.lessons_data:
            if self.current_exercise_index > 0:
                self.display_exercise(self.current_exercise_index - 1)
            else:
                self.output_text_edit.append("Already on the first exercise of this lesson.")

    def next_lesson(self):
        if self.lessons_data and self.current_lesson_index < len(self.lessons_data) - 1:
            self.load_lesson(self.current_lesson_index + 1)
        else:
            self.output_text_edit.append("Already on the last lesson.")

    def prev_lesson(self):
        if self.current_lesson_index > 0:
            self.load_lesson(self.current_lesson_index - 1)
        else:
            self.output_text_edit.append("Already on the first lesson.")

    def update_score_label(self, text):
        self.score_label.setText(text)

    def update_lesson_list_widget(self):
        pass
    
    def apply_syntax_highlighting(self):
        pass

    def run_user_code(self):
        self.output_text_edit.append("\n--- Running User Code ---")
        user_code = self.code_editor.toPlainText()
        output_buffer = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = output_buffer
        sys.stderr = output_buffer

        try:
            exec(user_code, {'__builtins__': builtins})
        except Exception as e:
            self.output_text_edit.append(f"Execution Error: {e}")
            self.output_text_edit.append(f"\nTraceback:\n{traceback.format_exc()}")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            self.output_text_edit.append(f"\nOutput:\n{output_buffer.getvalue().strip()}")
            self.output_text_edit.append("\n--------------------------")

    def ask_ai_general_help(self):
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
        self.ai_enabled = self.settings.value("ai_enabled", True, type=bool)
        logging.info(f"Loaded AI enabled setting: {self.ai_enabled}")
        self.update_ai_status_label()
        
        if hasattr(self, 'toggle_ai_action'):
            self.toggle_ai_action.setChecked(self.ai_enabled)
            self.toggle_ai_action.setText(f"AI Features: {'Enabled' if self.ai_enabled else 'Disabled'}")

    def save_settings(self):
        """Saves application settings, including AI enabled state."""
        # QSettings can save bools directly.
        self.settings.setValue("ai_enabled", self.ai_enabled)
        logging.info(f"Saved AI enabled setting: {self.ai_enabled}")

    def update_ai_status_label(self):
        if self.ai_enabled_globally:
            if self.ai_enabled and self.ai_agent and self.ai_agent.api_status == "READY":
                self.ai_status_label.setText("AI Status: <font color='green'>Ready</font>")
            elif self.ai_enabled and self.ai_agent and self.ai_agent.api_status != "READY":
                 self.ai_status_label.setText(f"AI Status: <font color='orange'>Not Ready ({self.ai_agent.api_status})</font>")
            elif not self.ai_enabled:
                self.ai_status_label.setText("AI Status: <font color='red'>Disabled (User)</font>")
        else:
            self.ai_status_label.setText("AI Status: <font color='red'>Globally Disabled (No API Key/Error)</font>")

    def toggle_ai_features(self):
        if not self.ai_enabled_globally:
            QMessageBox.warning(self, "AI Disabled", "AI features are globally disabled (e.g., no API key or init error) and cannot be toggled.")
            self.toggle_ai_action.setChecked(False) # Ensure checkbox stays unchecked
            return

        self.ai_enabled = self.toggle_ai_action.isChecked()
        self.settings.setValue("ai_enabled", self.ai_enabled)
        self.update_ai_status_label()
        if self.ai_enabled:
            self.output_text_edit.append("\nAI features enabled.")
            QMessageBox.information(self, "AI Enabled", "AI features are now enabled.")
        else:
            self.output_text_edit.append("\nAI features disabled.")
            QMessageBox.information(self, "AI Disabled", "AI features are now disabled.")
        
        self.ask_ai_general_help_button.setEnabled(self.ai_enabled)
        self.check_answer_button.setEnabled(self.ai_enabled) # Assuming check_answer uses AI

    def show_about_dialog(self):
        QMessageBox.about(self, "About Python Learning Tool",
                          "<h2>Python Learning Tool v1.0</h2>"
                          "<p>A desktop application for learning Python with AI assistance.</p>"
                          "<p>Developed with PySide6.</p>")

    def apply_window_box_styles(self):
        """Applies a dark theme and rounded corners to the main window's boxes."""
        style_sheet = """
        QMainWindow {
            background-color: #2b2b2b;
            color: #f0f0f0;
        }
        QTextEdit, QListWidget {
            background-color: #3c3c3c;
            color: #f0f0f0;
            border: 1px solid #555;
            border-radius: 5px;
            padding: 5px;
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
        QLabel {
            color: #f0f0f0;
        }
        QSplitter::handle {
            background-color: #505050;
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
            border: 1px solid #555;
            color: #f0f0f0;
        }
        QMenu::item {
            padding: 5px 20px;
        }
        QMenu::item:selected {
            background-color: #555;
        }
        QStatusBar {
            background-color: #3c3c3c;
            color: #f0f0f0;
            border-top: 1px solid #555;
        }
        #lessonList { /* Object name for specific styling */
            background-color: #333;
            border: 1px solid #444;
        }
        #lessonList::item:selected {
            background-color: #0078d7; /* Highlight color for selected item */
            color: #ffffff;
        }
        #lessonContent, #codeEditor, #outputText, #aiQuestionInput {
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 10pt;
        }
        """
        self.setStyleSheet(style_sheet)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    tool = PythonLearningTool()
    tool.show()
    sys.exit(app.exec())
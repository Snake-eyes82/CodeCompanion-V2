# CodeCompanion V2: Interactive Python Learning Tool with AI

## Table of Contents
- [About CodeCompanion V2](#about-codecompanion-v2)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [API Key Setup](#api-key-setup)
  - [Running the Application](#running-the-application)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## About CodeCompanion V2

CodeCompanion V2 is a desktop-based interactive Python learning tool designed to help beginners and intermediate users enhance their Python programming skills. It provides a structured learning environment with integrated lessons, a live code editor, and AI assistance to guide users through their coding journey. This version focuses on improved UI stability, refined AI integration, and a more robust lesson management system.

## Features

* **Interactive Lesson Display:** Presents Python lessons in a readable format, guiding users through concepts.
* **Integrated Code Editor:** A built-in editor allows users to write and test their Python code directly within the application.
* **Code Execution:** Run user-written Python code and display the output in a dedicated console area.
* **Exercise Checking (AI-Enhanced):** Evaluate user code against predefined exercise criteria, with optional AI feedback for guidance.
* **AI Integration:**
    * Leverages the Google Gemini API for powerful AI capabilities.
    * **"Ask AI for General Help" Feature:** Get instant explanations, syntax help, or conceptual clarification from the AI.
    * **Initial Lesson Generation:** If no `lessons.json` exists, the AI can generate a set of foundational Python lessons to get started.
* **User Progress Tracking:** Saves and loads user's progress through lessons and exercises, ensuring continuity.
* **Intuitive Navigation:** Easily switch between lessons and exercises using dedicated navigation buttons.
* **Customizable UI:** Features a dark theme for comfortable coding sessions.
* **Syntax Highlighting:** (Planned/Implemented) Improves code readability in the editor.

## Getting Started

Follow these instructions to set up and run CodeCompanion V2 on your local machine.

### Prerequisites

* Python 3.8+
* `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_GITHUB_USERNAME/CodeCompanion-V2.git](https://github.com/YOUR_GITHUB_USERNAME/CodeCompanion-V2.git)
    cd CodeCompanion-V2
    ```
    *(Replace `YOUR_GITHUB_USERNAME` with your actual GitHub username if you're cloning your own repo after pushing it.)*

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### API Key Setup

CodeCompanion V2 utilizes the Google Gemini API for its AI features. You need to obtain an API key and set it as an environment variable.

1.  **Get your Google Gemini API Key:**
    * Go to [Google AI Studio](https://aistudio.google.com/ "Google AI Studio") or the Google Cloud Console.
    * Create a new API key.

2.  **Set the API Key as an Environment Variable:**

    * **For Windows (PowerShell):**
        ```powershell
        $env:GOOGLE_API_KEY="YOUR_API_KEY_HERE"
        # To make it permanent (requires system restart or logging out/in):
        # Go to System Properties > Environment Variables.
        # Under "User variables" or "System variables", click "New...".
        # Variable name: GOOGLE_API_KEY
        # Variable value: YOUR_API_KEY_HERE
        ```

    * **For macOS/Linux (Bash/Zsh):**
        ```bash
        export GOOGLE_API_KEY="YOUR_API_KEY_HERE"
        # To make it permanent, add the above line to your shell's profile file (e.g., ~/.bashrc, ~/.zshrc)
        # Then run: source ~/.bashrc (or ~/.zshrc)
        ```
    * **Important:** Replace `"YOUR_API_KEY_HERE"` with your actual API key. The API key set via `export` or `$env:` is only for the current terminal session. For persistent use, set it as a system-wide environment variable.

### Running the Application

After installing dependencies and setting up your API key:

```bash
python main.py

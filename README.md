# CodeCompanion V2: Interactive Python Learning Tool with AI

## Table of Contents
- [CodeCompanion V2: Interactive Python Learning Tool with AI](#codecompanion-v2-interactive-python-learning-tool-with-ai)
  - [Table of Contents](#table-of-contents)
  - [🚀 Project Overview](#-project-overview)
  - [✨ Vision \& Long-Term Goals](#-vision--long-term-goals)
  - [💡 Why a Partner?](#-why-a-partner)
  - [🛠️ Tech Stack](#️-tech-stack)
  - [✅ Features](#-features)
  - [🚀 Getting Started](#-getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [API Key Setup (For Developers \& Advanced Users)](#api-key-setup-for-developers--advanced-users)
    - [Running the Application](#running-the-application)

---

## 🚀 Project Overview

CodeCompanion V2 is a desktop-based interactive Python learning tool designed to help beginners and intermediate users enhance their Python programming skills. It provides a structured learning environment with integrated lessons, a live code editor, and AI assistance to guide users through their coding journey. This version focuses on improved UI stability, refined AI integration, and a more robust lesson management system, all built with an eye towards future expandability and a market-ready release.

## ✨ Vision & Long-Term Goals

Our ambition is to evolve CodeCompanion into a comprehensive, multi-language learning environment, leveraging cutting-edge AI for personalized feedback and content generation. Our key aspirations include:

* **Multi-Language Support:** Expand beyond Python to offer interactive lessons and coding environments for other popular languages (e.g., JavaScript, C#, Java, C++).
* **Hybrid AI Model:** Implement a tiered AI access system. We plan to offer a foundational free tier powered by a developer-provided AI key (for easy user onboarding), alongside a premium or pro option where users can integrate their own API keys for extended usage or advanced features.
* **Enhanced Interactivity:** Develop more sophisticated interactive exercises, quizzes, and project-based learning modules.
* **Polished User Experience:** Refine the UI/UX, add robust error handling, and optimize performance for a seamless and intuitive user experience.

## 💡 Why a Partner?

I'm seeking a passionate and skilled co-developer to join me on this exciting journey. This project has a solid working prototype, and I'm looking for someone to:

* **Accelerate Feature Development:** Collaborate on implementing new UI elements and core functionalities.
* **Contribute to Architecture & Design:** Participate in decisions for scalable and maintainable code.
* **Expand Language Support:** Potentially specialize in integrating new programming languages into the platform.
* **Prepare for Market Release:** Assist in polishing the application, improving stability, and preparing for distribution.

If you're a Python developer with a strong interest in desktop applications, AI, and educational software, and are looking for a collaborative, shared-ownership project with significant growth potential, I'd love to connect!

## 🛠️ Tech Stack

* **Python:** Core programming language.
* **PySide6:** For the cross-platform desktop GUI.
* **Google Gemini API:** For powerful AI capabilities.
* **QSettings:** For persistent application settings (including user-provided API keys).
* **python-dotenv:** (For developers only) Manages local environment variables for API keys during development.

## ✅ Features

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
* **Customizable UI:** Features a dark theme for comfortable coding sessions (with plans for more customization).
* **Syntax Highlighting:** Improves code readability in the editor.

## 🚀 Getting Started

Follow these instructions to set up and run CodeCompanion V2 on your local machine.

### Prerequisites

* Python 3.8+
* `pip` (Python package installer)
* `git` (for cloning the repository)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_GITHUB_USERNAME/CodeCompanionV2.git](https://github.com/YOUR_GITHUB_USERNAME/CodeCompanionV2.git)
    cd CodeCompanionV2
    ```
    *(**Important:** Replace `YOUR_GITHUB_USERNAME` with your actual GitHub username. Make sure the repo name `CodeCompanionV2` matches exactly if you rename your GitHub repo from `CodeCompanion-V2`)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(If you haven't already, generate this file by running `pip freeze > requirements.txt` in your activated virtual environment.)*

### API Key Setup (For Developers & Advanced Users)

CodeCompanion V2 utilizes the Google Gemini API for its AI features.

1.  **Get your Google Gemini API Key:**
    * Go to [Google AI Studio](https://aistudio.google.com/app/apikey) or the Google Cloud Console.
    * Create a new API key.

2.  **Developer Convenience (using `.env` file):**
    * For local development, you can create a file named `.env` in the root of your project (the same directory as `main.py`).
    * Add your API key to the `.env` file:
        ```
        GOOGLE_API_KEY=YOUR_ACTUAL_GOOGLE_API_KEY_HERE
        ```
        *(Replace `YOUR_ACTUAL_GOOGLE_API_KEY_HERE` with your actual key. Ensure this `.env` file is listed in your `.gitignore`.)*

3.  **For End-Users (in-app settings):**
    * Once the application is running, navigate to the **Settings** menu.
    * You will find a dedicated field to paste your Google API Key. The application will securely save this key for future sessions using `QSettings`. This is the primary method for end-users and the key will persist across application launches.

### Running the Application

After installing dependencies and setting up your API key:

```bash
python main.py
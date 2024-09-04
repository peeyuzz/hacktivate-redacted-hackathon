(i dont want to do this im tired af but anyways here we go...)
# Hacktivate Redacted Hackathon

A Python-based tool for detecting and redacting sensitive information (such as faces or text) from various files, generating output in different formats.

## Project Structure

```
Hacktivate-Redacted-Hackathon
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── redactor.py
│   └── utils.py
│
├── model/
│   └── public/ultra-lightweight-face-detection-rfb/
│       ├── FP16/
│       └── ultra-lightweight-face-detection-rfb/
│
├── processed/  # Folder for processed files
│
├── static/
│   └── redacted_files/  # Folder for storing redacted files
│
├── templates/
│   └── index.html  # HTML template for the web interface
│
├── tests/  # Folder for unit tests
│
├── .env  # Environment variables file
├── .gitignore  # Git ignore file
├── beep.wav  # Sound file used in the project
├── kernel.errors.txt  # Log file for kernel errors
├── prompt.txt  # Prompt file for guidance
├── README.md  # Project documentation
├── requirements.txt  # Python dependencies
├── run_uvicorn.ps1  # Powershell script to run the server
└── TODO.txt  # List of pending tasks
```

## Prerequisites

- **Python**: Make sure Python is installed on your system.
- **Tesseract OCR**: Installed and configured correctly for text detection.
- **FFmpeg**: Required for handling media files.

### Installing Tesseract OCR

#### Windows
1. Download the Tesseract OCR installer from the [official Tesseract GitHub repository](https://github.com/tesseract-ocr/tesseract).
2. Run the installer and follow the setup instructions.
3. Add the installation path (e.g., `C:\Program Files\Tesseract-OCR`) to your system's PATH environment variable:
   - Right-click on "This PC" > "Properties" > "Advanced system settings" > "Environment Variables".
   - Find the "Path" variable, click "Edit", and add the path to the Tesseract OCR directory.

#### macOS
- Install Tesseract OCR using Homebrew:
  ```sh
  brew install tesseract
  ```

#### Linux
- Install Tesseract OCR using your package manager:
  ```sh
  sudo apt-get install tesseract-ocr
  ```

### Installing FFmpeg

#### Windows
1. Download the latest FFmpeg build from the [FFmpeg website](https://ffmpeg.org/download.html).
2. Extract the downloaded files to a folder.
3. Add the bin folder (e.g., `C:\ffmpeg\bin`) to your system's PATH environment variable:
   - Right-click on "This PC" > "Properties" > "Advanced system settings" > "Environment Variables".
   - Find the "Path" variable, click "Edit", and add the path to the FFmpeg bin directory.

#### macOS
- Install FFmpeg using Homebrew:
  ```sh
  brew install ffmpeg
  ```

#### Linux
- Install FFmpeg using your package manager:
  ```sh
  sudo apt-get install ffmpeg
  ```

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/peeyuzz/hacktivate-redacted-hackathon.git
   cd hacktivate-redacted-hackathon
   ```

2. Install Python dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Running the Application

To start the application, run the following command using PowerShell:

```sh
./run_uvicorn.ps1
```

This script will start the server, and the application will be accessible via your localhost.

## Configuration

- **Environment Variables**: Store any sensitive keys or configuration details in the `.env` file.
- **Templates and Static Files**: Customize the `index.html` and other assets located in the `templates` and `static` folders to match your application's needs.

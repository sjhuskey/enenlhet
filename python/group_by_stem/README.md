# Organize Audio + Annotation Files Script

This script automatically groups related files—`.wav`, `.eaf`, and `.trs`—into their own folders based on shared filename stems.  
It is designed for linguistics and documentation workflows where each recording session has multiple associated files.

Example:

```
MyRecording_2011-07-18.wav
MyRecording_2011-07-18.eaf
MyRecording_2011-07-18.trs
```

After running the script, these will be placed together in:

```
MyRecording_2011-07-18/
    MyRecording_2011-07-18.wav
    MyRecording_2011-07-18.eaf
    MyRecording_2011-07-18.trs
```

## Features

- Automatically finds matching `.wav`, `.eaf`, `.trs` by base name  
- Groups files into correctly named folders  
- Handles edited versions correctly (e.g., `_ed-2019-11-04`)  
- Produces a CSV log (`move_log.csv`)  
- Works on macOS and Linux  
- Uses only Python’s standard library  

## Installation

### 1. Create a folder for personal command-line scripts

```bash
mkdir -p "$HOME/python_scripts"
```

### 2. Copy the script into that folder

Place `group_by_stem.py` in:

```
~/python_scripts/
```

### 3. Add that folder to your PATH

```bash
echo 'export PATH="$HOME/python_scripts:$PATH"' >> ~/.zshrc
echo 'export PATH="$HOME/python_scripts:$PATH"' >> ~/.bash_profile
```

Restart Terminal afterward.

### 4. Make the script executable

```bash
chmod +x ~/python_scripts/group_by_stem.py
```

## Usage

Navigate to your data directory:

```bash
cd /path/to/data
```

Run the tool:

```bash
group_by_stem.py .
```

You can also specify the target directory:

```bash
group_by_stem.py /path/to/directory
```

If you want to do a dry run without making any changes, add `--dry-run` to the command:

```bash
group_by_stem.py /path/to/directory --dryrun
```

Result: grouped folders and a `move_log.csv` in the target directory.

## Troubleshooting

- **Command not found**  
  Run: `source ~/.zshrc` or `source ~/.bash_profile`

- **Files not grouped**  
  Ensure filenames match exactly up to their base stem, excluding `_ed-*` variants.

## Questions?

Feel free to ask for enhancements or debugging help.

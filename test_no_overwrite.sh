#!/bin/bash
# Test script to demonstrate the --no-overwrite functionality

echo "🧪 Testing --no-overwrite functionality for audiobook generators"
echo

# Create test directories
mkdir -p test_demo/input test_demo/output

# Create test text files
cat > test_demo/input/chapter1.txt << 'EOF'
[narrator]Welcome to chapter one of our test audiobook.
[character1]Hello there! I'm excited to be here.
[narrator]The character spoke with enthusiasm.
...
[narrator]After a pause, the story continued.
EOF

cat > test_demo/input/chapter2.txt << 'EOF'
[narrator]This is chapter two.
[character2]I have something important to say!
[narrator]And that concludes our test.
EOF

# Create voice configuration
cat > test_demo/voice_config.json << 'EOF'
{
  "narrator": {
    "temperature": 0.7,
    "top_p": 0.8,
    "output_format": "wav"
  },
  "character1": {
    "temperature": 0.9,
    "output_format": "wav"
  },
  "character2": {
    "temperature": 0.8,
    "output_format": "wav"
  },
  "default": {
    "temperature": 0.8,
    "output_format": "wav"
  }
}
EOF

# Create some existing files to test the no-overwrite functionality
touch test_demo/output/chapter1_000.wav
touch test_demo/output/chapter1_002.wav
touch test_demo/output/chapter2_001.wav

echo "📁 Created test setup:"
echo "  - Input files: chapter1.txt, chapter2.txt"
echo "  - Voice config: voice_config.json"
echo "  - Pre-existing WAV files:"
ls -la test_demo/output/

echo
echo "🔧 Test 1: Show help for improved version"
python improved_audiobook_generator.py --help

echo
echo "🔧 Test 2: Show help for basic version"
python audiobook_generator.py --help

echo
echo "📝 Created test files in test_demo/ directory"
echo "   You can run the following commands to test:"
echo
echo "   # Test normal operation (will attempt to generate all files)"
echo "   python improved_audiobook_generator.py test_demo/input test_demo/output test_demo/voice_config.json"
echo
echo "   # Test with --no-overwrite (will skip existing files)"
echo "   python improved_audiobook_generator.py test_demo/input test_demo/output test_demo/voice_config.json --no-overwrite"
echo
echo "   # Test with other options"
echo "   python improved_audiobook_generator.py test_demo/input test_demo/output test_demo/voice_config.json --no-overwrite --retries 5"
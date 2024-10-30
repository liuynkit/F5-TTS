curl -X POST "http://192.168.0.64:5053/tts/infer/e2tts" \
     -F "text=Kleidung macht Leute, sagt die Katze." \
     -F "speed=1.0" \
     -F "iteration_steps=10" \
     -o response.json

# Extract and decode the audio
audio_base64=$(jq -r '.audio' response.json)
echo $audio_base64 | base64 -d > audio.raw

# Convert raw audio to WAV (assuming 24000 Hz sample rate and 16-bit PCM)
ffmpeg -f s16le -ar 24000 -ac 1 -i audio.raw -acodec pcm_f32le /project/tts/students/yining_ws/multi_lng/F5-TTS/outputs/worker_output.wav

# Clean up temporary files
rm response.json audio.raw
import sys
from audiocraft.models import MusicGen
from audiocraft.utils import audio_write

def main():
    # Parse user inputs (for simplicity)
    prompt = input("Enter a music description: ")
    duration = int(input("Enter duration in seconds: "))
    output_file = input("Enter output filename (without extension): ") + ".wav"

    # Load the pretrained model
    print("Loading MusicGen model...")
    model = MusicGen.get_pretrained("melody")  # Use "small", "medium", "large", or "melody"

    # Set generation parameters
    model.set_generation_params(duration=duration)

    # Generate music
    print(f"Generating music: '{prompt}'")
    generated_audio = model.generate([prompt])

    # Save the output to a file
    audio_write(output_file, generated_audio[0], model.sample_rate)
    print(f"Music saved as: {output_file}")

if __name__ == "__main__":
    main()


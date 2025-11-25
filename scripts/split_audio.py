import os
import argparse
from pydub import AudioSegment
from pydub.utils import make_chunks
import glob 

def main(args):

    os.makedirs(args.save_path, exist_ok=True)
    search_pattern = os.path.join(args.input_folder, '**', '*.wav')
    audio_files = glob.glob(search_pattern, recursive=True)

    total_chunks_created = 0
    chunk_length_ms = 1000 

    for file_path in audio_files:
        try:
            audio = AudioSegment.from_file(file_path) 

            if args.sample_rate:
                audio = audio.set_frame_rate(args.sample_rate)

            chunks = make_chunks(audio, chunk_length_ms)

            for i, chunk in enumerate(chunks):
                if args.num_chunks is not None and total_chunks_created >= args.num_chunks:
                    print(f"\nReached the limit of {args.num_chunks} chunks.")
                    return 
                current_chunk_index = args.start_name + total_chunks_created
                chunk_name = f"background_{current_chunk_index}.wav"
                chunk_save_path = os.path.join(args.save_path, chunk_name)

                if len(chunk) > 0: 
                   chunk.export(chunk_save_path, format="wav")
                   total_chunks_created += 1
                else:
                   print(f"Skipping empty chunk from {file_path}")

                if total_chunks_created % 100 == 0:
                     print(f"Created {total_chunks_created} chunks...", end='\r')


        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    print(f"\nFinished processing. Total chunks created: {total_chunks_created}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to split audio files from a folder into 1-second chunks.")

    parser.add_argument('--input_folder', type=str, required=True,
                        help='Path to the folder containing input audio files (.wav).')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Full path to the folder where chunked audio will be saved.')
    parser.add_argument('--num_chunks', type=int, default=None,
                        help='Maximum total number of chunks to generate. If None, process all files.')
    parser.add_argument('--start_name', type=int, default=0,
                        help='Starting index for the output chunk file names (e.g., background_START_NAME.wav).')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Resample audio to this rate before chunking (e.g., 16000). If None, keeps original.')

    args = parser.parse_args()

    main(args)

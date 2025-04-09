#date: 2025-04-09T16:59:17Z
#url: https://api.github.com/gists/b311cd6f4d4a4a97b0ba5d3ffbffac67
#owner: https://api.github.com/users/robbie-anam

import os
import time

import dotenv
from cartesia import Cartesia, WordTimestamps

random_transcript = (
    "Hi there is there anything I can help you with today. You are looking really great today. Whats that I hear about you being a great person. "
    "Thats really interesting but it's not for me to say, or is it. I don't really know to tell the truth. "
    "Here is some nonsensical ramblings: the house is a house and the car is a car. The cat is a cat and the dog is a dog. The fish is a fish and the bird is a bird. "
    "The cow is a cow and the horse is a horse. The pig is a pig and the sheep is a sheep. "
    "The goat is a goat and the chicken is a chicken. The duck is a duck and the turkey is a turkey. "
    "The rabbit is a rabbit and the hamster is a hamster. The turtle is a turtle and the lizard is a lizard."
    " The snake is a snake and the frog is a frog. The mouse is a mouse and the rat is a rat. Potato.")


def generate_words_at_rate(chars_per_second: int = 1000):
    """
    Generate words at a rate of chars_per_second.
    """
    transcript_split_into_sentences = random_transcript.split(". ")
    transcript_split_into_sentences = [s + "." for s in transcript_split_into_sentences]
    for sentence in transcript_split_into_sentences:
        time_to_sleep = len(sentence) / chars_per_second
        time.sleep(time_to_sleep)
        yield sentence


if __name__ == "__main__":
    # For testing purposes only
    dotenv.load_dotenv()
    client = Cartesia(api_key=os.getenv("CARTESIA_API_KEY"))

    ws = client.tts.websocket()
    voice_config = {
        "mode": "id",
        "id": "7e19344f-9f17-47d7-a13a-4366ad06ebf3",  # Gabe
        "__experimental_controls": {"speed": 1.0},
    }

    result = ws.context("TEST_CONTEXT_ID").send(
        model_id="sonic-2",
        transcript=generate_words_at_rate(chars_per_second=10),
        voice=voice_config,
        output_format={"container": "raw", "encoding": "pcm_s16le", "sample_rate": 16_000},
        add_timestamps=True,
        stream=True,
    )
    raw_data = b""
    start = time.time()
    first_audio_recevied_at = None
    first_word_timing_received_at = None
    audio_at = 0.0
    words_at = 0.0
    count_words_ahead = 0
    count_audio_ahead = 0
    total_chunks = 0
    for chunk in result:
        received_at = time.time()
        if chunk.audio is not None:
            audio = chunk.audio
            raw_data += audio
            if first_audio_recevied_at is None:
                first_audio_recevied_at = received_at
            audio_at = len(raw_data) / (16000 * 2 * 1)
        if chunk.word_timestamps is not None:
            word_timestamps_obj: WordTimestamps = chunk.word_timestamps
            print(word_timestamps_obj.words, word_timestamps_obj.start, word_timestamps_obj.end)
            if first_word_timing_received_at is None:
                first_word_timing_received_at = received_at
            if min(word_timestamps_obj.start) < words_at:
                print("BUG: Word timings have gone backwards!")
            words_at = max(word_timestamps_obj.end)

        print(f"Audio at: {audio_at:.2f}s. Word at: {words_at:.2f}s")
        total_chunks += 1
        if audio_at > words_at:
            count_audio_ahead += 1
        elif words_at > audio_at:
            count_words_ahead += 1

    print(
        f"First audio received at: {first_audio_recevied_at - start:.2f}s. First word received at: {first_word_timing_received_at - start:.2f}s")
    print(f"Total chunks: {total_chunks}. Audio ahead: {count_audio_ahead}. Words ahead: {count_words_ahead}")

    print(f"DONE!")

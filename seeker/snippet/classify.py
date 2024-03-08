#date: 2024-03-08T17:04:23Z
#url: https://api.github.com/gists/c22e4b72a166f7fd0476c0512b7e5482
#owner: https://api.github.com/users/mvoodarla

import sieve
import os
import cv2

moondream = sieve.function.get("sieve/moondream")

def create_question(client, thing):
    completion = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {
                "role": "system",
                "content": """Format whatever you recieve into a question that's fed into an image model. For example, if you recieve "grass color", you should format it into a question like "What is the color of the grass?" and send it to the model. Make sure the question prompts for detail.""",
            },
            {
                "role": "user",
                "content": f"""i am trying to understand the following thing about an image: {thing}""",
            },
        ],
    )
    return completion.choices[0].message.content

def vote_on_answer(client, thing, answers):
    answers = "\n".join([f"{i+1}. {ans}" for i, ans in enumerate(answers)])
    completion = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {
                "role": "system",
                "content": f"""You are trying to classify '{thing}' about an image or video based on some responses you are fed. Reply with your best answer. Don't say anything else.""",
            },
            {
                "role": "user",
                "content": f"""{answers}""",
            },
        ],
    )
    return completion.choices[0].message.content

@sieve.function(
    name="classify",
    python_version="3.10",
    system_packages=["libgl1-mesa-glx", "libglib2.0-0", "ffmpeg"],
    python_packages=[
        "opencv-python",
        "decord",
        "openai",
    ],
    environment_variables=[
        sieve.Env(name="OPENAI_API_KEY", description="OpenAI API key")
    ],
)
def classify(file: sieve.File, thing: str = "type of content"):
    """
    :param file: image or video to classify
    :param thing: what to classify about the image or video
    """
    from openai import OpenAI
    client = OpenAI()

    question = create_question(client, thing)
    video_extensions = ["mp4", "avi", "mov", "flv"]
    image_extensions = ["jpg", "jpeg", "png", "bmp", "tiff"]

    file_path = file.path
    results = []
    if file_path.split(".")[-1] in video_extensions:
        from decord import VideoReader
        vr = VideoReader(file_path)
        num_frames = len(vr)

        if num_frames > 10:
            # pick 10 evenly spaced frames within the video
            frames = [int(num_frames / 10 * i) for i in range(1, 11)]
            frames = [f for f in frames if f < num_frames]
        else:
            frames = [0]

        os.makedirs("frames", exist_ok=True)
        answers = []
        for i, f in enumerate(frames):
            frame = vr[f].asnumpy()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (384, 384))
            cv2.imwrite(f"frames/frame{i}.jpg", frame)
            answer = moondream.push(sieve.File(path=f"frames/frame{i}.jpg"), question=question)
            answers.append(answer)

        results = [answer.result() for answer in answers]


    elif file_path.split(".")[-1] in image_extensions:
        answer = moondream.push(file, question=question)
        results = [answer.result()]
    
    if len(results) == 0:
        raise ValueError("File type not supported. Only images and videos are supported.")
    
    best_answer = vote_on_answer(client, thing, results)
    return best_answer
    
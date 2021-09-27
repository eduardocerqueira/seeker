#date: 2021-09-27T17:16:08Z
#url: https://api.github.com/gists/eef5104fd88ab64f724367159f2ddb74
#owner: https://api.github.com/users/mfrashad

#@title API Functions
import clipit
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import FileResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# define function to be run as background tasks
def generate(email, settings):
    clipit.do_init(settings)
    clipit.do_run(settings)

    prompt = " | ".join(settings.prompts)

    email_results_mailgun(email, prompt)

@app.get('/')
async def root():
    return {'hello': 'world'}

@app.post("/generate")
async def add_task(
        email: str,
        background_tasks: BackgroundTasks,
        seed: int = Form(None),
        iterations: int = Form(None),
        prompts: str = Form("Underwater City"),
        quality: str = Form("draft"),
        aspect: str = Form("square"),
        scale: float = Form(2.5),
        style: str = Form('image'),
        make_video: bool = Form(False),      
    ):
    torch.cuda.empty_cache()
    clipit.reset_settings()

    use_pixeldraw = (style == 'Pixel Art')
    use_clipdraw = (style == 'Painting')
    clipit.add_settings(prompts=prompts,
                        seed=seed,
                        iterations=iterations,
                        aspect=aspect,
                        quality=quality,
                        scale=scale,
                        use_pixeldraw=use_pixeldraw,
                        use_clipdraw=use_clipdraw,
                        make_video=make_video)
    
    settings = clipit.apply_settings()

    # Run function as background task
    background_tasks.add_task(generate, email, settings)

    return {"message": "Task is processed in the background"}
  
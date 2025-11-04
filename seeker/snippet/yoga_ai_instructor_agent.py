#date: 2025-11-04T17:11:20Z
#url: https://api.github.com/gists/9cb2c5695803c6ebab8a3a8044125957
#owner: https://api.github.com/users/amosgyamfi

import asyncio
import logging
from uuid import uuid4

from dotenv import load_dotenv

from vision_agents.core.edge.types import User
from vision_agents.core import agents
from vision_agents.plugins import getstream, ultralytics, gemini, openai

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Read the yoga instructor guide
with open("yoga_instructor_guide.md", "r") as f:
    YOGA_INSTRUCTOR_INSTRUCTIONS = f.read()


async def start_yoga_instructor() -> None:
    """
    Start the yoga instructor agent with real-time pose detection
    """
    logger.info("🧘 Starting Yoga AI Instructor Agent...")
    
    # Create the agent with YOLO pose detection
    agent = agents.Agent(
        edge=getstream.Edge(),  # Stream's edge for low-latency video transport
        agent_user=User(name="AI Yoga Instructor", id="yoga_instructor_agent"),
        instructions="Read @yoga_instructor_guide.md",
        
        # Choose your LLM - uncomment your preferred option:
        # Option 1: Gemini Realtime (good for vision analysis)
        llm=gemini.Realtime(),
        
        # Option 2: OpenAI Realtime (alternative option)
        # llm=openai.Realtime(fps=10, model="gpt-4o-realtime-preview"),
        
        # Add YOLO pose detection processor
        processors=[
            ultralytics.YOLOPoseProcessor(
                model_path="../../yolo11n-pose.pt",  # YOLO pose detection model
                conf_threshold=0.5,  # Confidence threshold for detection
                enable_hand_tracking=True,  # Enable hand keypoint detection for detailed feedback
            )
        ],
    )
    
    logger.info("✅ Agent created successfully")
    
    # Create the agent user in the system
    await agent.create_user()
    logger.info("✅ Agent user created")
    
    # Create a call (room) for the video session
    call_id = str(uuid4())
    call = agent.edge.client.video.call("default", call_id)
    logger.info(f"✅ Call created with ID: {call_id}")
    
    # Open the demo UI in browser
    await agent.edge.open_demo(call)
    logger.info("🌐 Demo UI opened in browser")
    
    # Join the call and start the session
    logger.info("🎥 Agent joining the call...")
    with await agent.join(call):
        # Initial greeting and instructions
        await agent.llm.simple_response(
            text=(
                "Namaste! 🧘‍♀️ I'm your AI yoga instructor with a soft Scottish accent. "
                "I'll be guiding you through your practice today with the help of pose analysis. "
                "I can help you with standing poses, seated poses, transitions, and breathing. "
                "Just step onto your mat and show me what you'd like to work on. "
                "Remember to breathe, ground yourself, and listen to your body. Let's begin!"
            )
        )
        
        logger.info("🧘 Session active - providing real-time yoga feedback...")
        
        # Run until the call ends
        await agent.finish()
        
    logger.info("👋 Session ended - Namaste!")


if __name__ == "__main__":
    print("=" * 70)
    print("🧘  AI YOGA INSTRUCTOR  🧘")
    print("=" * 70)
    print("\n📋 Features:")
    print("  ✓ Real-time pose detection and alignment tracking")
    print("  ✓ Expert guidance on yoga asanas (poses)")
    print("  ✓ Personalized feedback on form and technique")
    print("  ✓ Breath synchronization coaching")
    print("  ✓ Safe transition guidance")
    print("\n🎯 Poses Supported:")
    print("  • Standing Poses (Mountain, Warrior, Tree, Triangle, etc.)")
    print("  • Seated Poses (Lotus, Pigeon, Forward Bends, etc.)")
    print("  • Balance Poses (Eagle, Half Moon, Standing Splits, etc.)")
    print("  • Flexibility Poses (Splits, Bound Angle, etc.)")
    print("\n🚀 Starting agent...\n")
    
    asyncio.run(start_yoga_instructor())
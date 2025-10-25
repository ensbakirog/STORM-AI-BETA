import os

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, inference
from livekit.plugins import noise_cancellation, silero, google
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env.local")

with open('storm_prompt.txt', 'r', encoding='utf-8') as f:
    STORM_INSTRUCTIONS = f.read()

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=STORM_INSTRUCTIONS,
        )


async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        stt=inference.STT(model="cartesia/ink-whisper", language="tr"),
        llm=inference.LLM(model="google/gemini-2.5-flash", extra_kwargs={"temperature": 1}),
        tts = google.beta.GeminiTTS(model="gemini-2.5-flash-preview-tts", voice_name="Kore"),
        vad=silero.VAD.load(activation_threshold=0.75, min_silence_duration=1.2, min_speech_duration=1),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` instead for best results
            noise_cancellation=noise_cancellation.BVC(), 
        ),
    )

    await session.generate_reply(
        instructions="Ben STORM Yapay Zeka Asistanıyım, nasıl yardımcı olabilirim?"
    )

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
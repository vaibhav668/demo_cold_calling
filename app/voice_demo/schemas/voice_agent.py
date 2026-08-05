import uuid
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class VoiceProfileOut(BaseModel):
    name: str
    description: Optional[str] = None
    avatar: Optional[str] = None
    gender: str
    supported_languages: str
    preview_audio: Optional[str] = None
    status: str

class SessionSetupIn(BaseModel):
    voice_name: str
    industry: str
    language: str

class SessionSetupOut(BaseModel):
    session_id: str
    campaign_id: uuid.UUID
    voice_profile: VoiceProfileOut

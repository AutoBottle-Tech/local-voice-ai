export interface SettingsOptions {
  llm_providers: string[];
  stt_providers: string[];
  tts_providers: string[];
  minimax_tts_models: string[];
  minimax_tts_model_labels: Record<string, string>;
  kokoro_voices: string[];
  minimax_voices: string[];
}

export interface AppSettingsConfig {
  llm_provider: string;
  llm_model: string;
  stt_provider: string;
  tts_provider: string;
  tts_voice: string;
  minimax_tts_model: string;
  minimax_tts_voice: string;
  wake_word: boolean;
  minimax_api_key_set: boolean;
  options: SettingsOptions;
}

export interface SettingsPatch {
  llm_provider?: string;
  stt_provider?: string;
  tts_provider?: string;
  tts_voice?: string;
  minimax_tts_model?: string;
  minimax_tts_voice?: string;
  wake_word?: boolean;
}

export const LLM_LABELS: Record<string, string> = {
  llama: 'Local Gemma (llama.cpp)',
  minimax: 'MiniMax M3 (cloud)',
};

export const STT_LABELS: Record<string, string> = {
  nemotron: 'Nemotron (local)',
  whisper: 'Whisper (local)',
};

export const TTS_LABELS: Record<string, string> = {
  kokoro: 'Kokoro (local)',
  minimax: 'MiniMax Speech (cloud)',
};

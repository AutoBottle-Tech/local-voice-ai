'use client';

import Link from 'next/link';
import { ArrowLeftIcon, CheckIcon, SpinnerIcon } from '@phosphor-icons/react/dist/ssr';
import { Button } from '@/components/livekit/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/livekit/select';
import { Toggle } from '@/components/livekit/toggle';
import { useSettings } from '@/hooks/useSettings';
import { LLM_LABELS, STT_LABELS, type SettingsPatch, TTS_LABELS } from '@/lib/settings';

function SettingRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
      <span className="text-foreground text-sm font-medium">{label}</span>
      <div className="sm:min-w-[240px]">{children}</div>
    </div>
  );
}

function ProviderSelect({
  value,
  options,
  labels,
  onChange,
  disabled,
}: {
  value: string;
  options: string[];
  labels: Record<string, string>;
  onChange: (value: string) => void;
  disabled?: boolean;
}) {
  return (
    <Select value={value} onValueChange={onChange} disabled={disabled}>
      <SelectTrigger className="w-full">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {options.map((option) => (
          <SelectItem key={option} value={option}>
            {labels[option] ?? option}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

function RestartProgress({
  services,
}: {
  services: { name: string; ready: boolean; detail?: string }[];
}) {
  return (
    <div className="border-border bg-muted/40 mt-6 rounded-2xl border p-4">
      <p className="text-foreground mb-3 text-sm font-medium">Applying changes…</p>
      <ul className="space-y-2">
        {services.map((child) => (
          <li key={child.name} className="text-foreground flex items-center gap-2 text-sm">
            {child.ready ? (
              <CheckIcon weight="bold" className="text-fg0 size-4" />
            ) : (
              <SpinnerIcon weight="bold" className="text-muted-foreground size-4 animate-spin" />
            )}
            <span className={child.ready ? '' : 'text-muted-foreground'}>{child.name}</span>
            {!child.ready && child.detail && (
              <span className="text-muted-foreground ml-auto font-mono text-xs tabular-nums">
                {child.detail}
              </span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

export function SettingsView() {
  const {
    config,
    draft,
    loading,
    error,
    saving,
    restarting,
    stackStatus,
    dirty,
    updateDraft,
    save,
  } = useSettings();

  if (loading || !config || !draft) {
    return (
      <div className="text-muted-foreground flex min-h-[50vh] items-center justify-center text-sm">
        Loading settings…
      </div>
    );
  }

  const voiceOptions =
    draft.tts_provider === 'minimax' ? config.options.minimax_voices : config.options.kokoro_voices;

  const activeVoice = draft.tts_provider === 'minimax' ? draft.minimax_tts_voice : draft.tts_voice;

  const handleTtsProviderChange = (provider: string) => {
    const patch: Partial<SettingsPatch> = { tts_provider: provider };
    if (provider === 'minimax') {
      patch.minimax_tts_voice = draft.minimax_tts_voice ?? config.options.minimax_voices[0] ?? '';
    } else {
      patch.tts_voice = draft.tts_voice ?? config.options.kokoro_voices[0] ?? '';
    }
    updateDraft(patch);
  };

  const handleVoiceChange = (voice: string) => {
    if (draft.tts_provider === 'minimax') {
      updateDraft({ minimax_tts_voice: voice });
    } else {
      updateDraft({ tts_voice: voice });
    }
  };

  return (
    <div className="mx-auto w-full max-w-lg px-6 pt-20 pb-28">
      <Link
        href="/"
        className="text-muted-foreground hover:text-foreground mb-8 inline-flex items-center gap-2 text-sm transition-colors"
      >
        <ArrowLeftIcon size={16} weight="bold" />
        Back to app
      </Link>

      <h1 className="text-foreground mb-2 text-2xl font-semibold tracking-tight">Settings</h1>
      <p className="text-muted-foreground mb-8 text-sm leading-6">
        Switch voice stack providers. Saving restarts affected services automatically.
      </p>

      <div className="space-y-6">
        <SettingRow label="Language model">
          <ProviderSelect
            value={draft.llm_provider ?? config.llm_provider}
            options={config.options.llm_providers}
            labels={LLM_LABELS}
            onChange={(value) => updateDraft({ llm_provider: value })}
            disabled={saving || restarting}
          />
        </SettingRow>

        <SettingRow label="Speech-to-text">
          <ProviderSelect
            value={draft.stt_provider ?? config.stt_provider}
            options={config.options.stt_providers}
            labels={STT_LABELS}
            onChange={(value) => updateDraft({ stt_provider: value })}
            disabled={saving || restarting}
          />
        </SettingRow>

        <SettingRow label="Text-to-speech">
          <ProviderSelect
            value={draft.tts_provider ?? config.tts_provider}
            options={config.options.tts_providers}
            labels={TTS_LABELS}
            onChange={handleTtsProviderChange}
            disabled={saving || restarting}
          />
        </SettingRow>

        {draft.tts_provider === 'minimax' && (
          <SettingRow label="TTS model">
            <ProviderSelect
              value={draft.minimax_tts_model ?? config.minimax_tts_model}
              options={config.options.minimax_tts_models}
              labels={config.options.minimax_tts_model_labels ?? {}}
              onChange={(value) => updateDraft({ minimax_tts_model: value })}
              disabled={saving || restarting}
            />
          </SettingRow>
        )}

        <SettingRow label="Voice">
          <Select
            value={activeVoice}
            onValueChange={handleVoiceChange}
            disabled={saving || restarting}
          >
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {voiceOptions.map((voice) => (
                <SelectItem key={voice} value={voice}>
                  {voice}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </SettingRow>

        <SettingRow label="Wake word">
          <Toggle
            pressed={!!draft.wake_word}
            onPressedChange={(pressed) => updateDraft({ wake_word: pressed })}
            disabled={saving || restarting}
            variant="secondary"
            className="w-full justify-center sm:w-auto"
          >
            {draft.wake_word ? 'On' : 'Off'}
          </Toggle>
        </SettingRow>

        <SettingRow label="MiniMax API key">
          <span
            className={
              config.minimax_api_key_set ? 'text-foreground text-sm' : 'text-destructive text-sm'
            }
          >
            {config.minimax_api_key_set ? 'Configured in .env' : 'Not set — add MINIMAX_API_KEY'}
          </span>
        </SettingRow>

        <p className="text-muted-foreground text-xs leading-5">
          Active LLM: <span className="text-foreground font-medium">{config.llm_model}</span>
        </p>
      </div>

      {error && (
        <p className="text-destructive mt-4 text-sm" role="alert">
          {error}
        </p>
      )}

      {restarting && stackStatus.children.length > 0 && (
        <RestartProgress services={stackStatus.children} />
      )}

      <div className="bg-background/95 fixed inset-x-0 bottom-0 border-t p-4 backdrop-blur">
        <div className="mx-auto flex max-w-lg justify-end">
          <Button
            variant="primary"
            size="lg"
            className="min-w-[160px] font-mono"
            disabled={!dirty || saving || restarting}
            onClick={() => void save()}
          >
            {saving || restarting ? 'Applying…' : 'Save changes'}
          </Button>
        </div>
      </div>
    </div>
  );
}

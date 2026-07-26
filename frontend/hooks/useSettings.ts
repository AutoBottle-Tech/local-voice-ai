'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import type { StackStatus } from '@/hooks/useStackStatus';
import type { AppSettingsConfig, SettingsPatch } from '@/lib/settings';

const POLL_INTERVAL_MS = 2000;

function configToDraft(config: AppSettingsConfig): SettingsPatch {
  return {
    llm_provider: config.llm_provider,
    stt_provider: config.stt_provider,
    tts_provider: config.tts_provider,
    tts_voice: config.tts_voice,
    minimax_tts_model: config.minimax_tts_model,
    minimax_tts_voice: config.minimax_tts_voice,
    wake_word: config.wake_word,
  };
}

async function fetchConfig(): Promise<AppSettingsConfig> {
  const res = await fetch('/api/config', { cache: 'no-store' });
  if (!res.ok) {
    throw new Error(`Failed to load settings (${res.status})`);
  }
  return res.json();
}

async function fetchStackStatus(): Promise<StackStatus> {
  const res = await fetch('/api/status', { cache: 'no-store' });
  if (!res.ok) {
    throw new Error(`Failed to load status (${res.status})`);
  }
  const data = await res.json();
  return {
    ready: data.ready,
    children: data.children,
    wakeWord: !!data.wake_word,
  };
}

export function useSettings() {
  const [config, setConfig] = useState<AppSettingsConfig | null>(null);
  const [draft, setDraft] = useState<SettingsPatch | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [restarting, setRestarting] = useState(false);
  const [stackStatus, setStackStatus] = useState<StackStatus>({
    ready: true,
    children: [],
    wakeWord: false,
  });

  const reload = useCallback(async () => {
    const next = await fetchConfig();
    setConfig(next);
    setDraft(configToDraft(next));
    return next;
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const next = await fetchConfig();
        if (cancelled) return;
        setConfig(next);
        setDraft(configToDraft(next));
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load settings');
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const dirty = useMemo(() => {
    if (!config || !draft) return false;
    const saved = configToDraft(config);
    return (Object.keys(saved) as (keyof SettingsPatch)[]).some((key) => saved[key] !== draft[key]);
  }, [config, draft]);

  const updateDraft = useCallback((patch: Partial<SettingsPatch>) => {
    setDraft((prev) => (prev ? { ...prev, ...patch } : prev));
  }, []);

  const waitForReady = useCallback(async () => {
    while (true) {
      const status = await fetchStackStatus();
      setStackStatus(status);
      if (status.ready) return;
      await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
    }
  }, []);

  const save = useCallback(async () => {
    if (!draft || !dirty) return;
    setSaving(true);
    setError(null);
    try {
      const res = await fetch('/api/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(draft),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? `Save failed (${res.status})`);
      }
      setRestarting(true);
      await waitForReady();
      await reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save settings');
    } finally {
      setSaving(false);
      setRestarting(false);
    }
  }, [draft, dirty, reload, waitForReady]);

  return {
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
  };
}

'use client';

import { useMemo } from 'react';
import { TokenSource } from 'livekit-client';
import {
  RoomAudioRenderer,
  SessionProvider,
  StartAudio,
  useSession,
} from '@livekit/components-react';
import type { AppConfig } from '@/app-config';
import { ViewController } from '@/components/app/view-controller';
import { Toaster } from '@/components/livekit/toaster';
import { useAgentErrors } from '@/hooks/useAgentErrors';
import { useDebugMode } from '@/hooks/useDebug';

const IN_DEVELOPMENT = process.env.NODE_ENV !== 'production';

function resolveAgentName(configAgent?: string): string | undefined {
  if (typeof window === 'undefined') return configAgent;
  const fromUrl = new URLSearchParams(window.location.search).get('agent');
  if (fromUrl === 'habits') return 'habits';
  return configAgent;
}

function buildTokenSource(agentName?: string) {
  return TokenSource.custom(async () => {
    const roomConfig = agentName ? { agents: [{ agent_name: agentName }] } : undefined;
    const res = await fetch('/api/connection-details', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ room_config: roomConfig }),
    });
    if (!res.ok) throw new Error('Error fetching connection details!');
    return res.json();
  });
}

function AppSetup() {
  useDebugMode({ enabled: IN_DEVELOPMENT });
  useAgentErrors();

  return null;
}

interface AppProps {
  appConfig: AppConfig;
}

export function App({ appConfig }: AppProps) {
  const agentName = useMemo(() => resolveAgentName(appConfig.agentName), [appConfig.agentName]);

  const tokenSource = useMemo(() => buildTokenSource(agentName), [agentName]);

  const session = useSession(tokenSource, agentName ? { agentName } : undefined);

  return (
    <SessionProvider session={session}>
      <AppSetup />
      <main className="grid h-svh grid-cols-1 place-content-center">
        <ViewController appConfig={appConfig} />
      </main>
      <StartAudio label="Start Audio" />
      <RoomAudioRenderer />
      <Toaster />
    </SessionProvider>
  );
}

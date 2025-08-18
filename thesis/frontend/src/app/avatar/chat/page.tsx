"use client";

import { NoAgentNotification } from "@/components/NoAgentNotification";
import TranscriptionView from "@/components/TranscriptionView";
import {
  BarVisualizer,
  RoomAudioRenderer,
  RoomContext,
  VideoTrack,
  useVoiceAssistant,
} from "@livekit/components-react";
import { AnimatePresence, motion } from "framer-motion";
import { Room, RoomEvent } from "livekit-client";
import { useCallback, useEffect, useState } from "react";
import {TalkingHeadComponent} from "@/components/TalkingHead/TalkingHeadComponent";
import { useAuth } from "@/contexts/AuthContext";
export interface ConnectionDetails {
  server_url: string;
  room_name: string;
  participant_name: string;
  participant_token: string;
}

export default function Page() {
  const [room] = useState(new Room());
  const { token: authToken } = useAuth(); // Get the auth token

  const onConnectButtonClicked = useCallback(async () => {
    if (!authToken) {
      console.error("Authentication token not found. Please login.");
      // Optionally, redirect to login or show a message
      return;
    }
    const url = new URL(
      process.env.NEXT_PUBLIC_BACKEND_URL + "/livekit/connection-details",
      window.location.origin
    );
    const response = await fetch(url.toString(), {
      headers: {
        'Authorization': `Bearer ${authToken}`, // Add the token to the header
      },
    });
    const connectionDetailsData: ConnectionDetails = await response.json();

    await room.connect(connectionDetailsData.server_url, connectionDetailsData.participant_token);
    await room.localParticipant.setMicrophoneEnabled(true);
  }, [room, authToken]); // Add authToken as a dependency

  useEffect(() => {
    room.on(RoomEvent.MediaDevicesError, onDeviceFailure);

    return () => {
      room.off(RoomEvent.MediaDevicesError, onDeviceFailure);
    };
  }, [room]);

  return (
    <div data-lk-theme="default" className="h-full grid content-center bg-[var(--lk-bg)]">
      <RoomContext.Provider value={room}>
        <div className="lk-room-container w-full mx-auto max-h-[90vh] min-h-[90vh]">
          <div className="flex flex-row h-full w-full" style={{justifyContent:'space-between',alignItems:'center'}}> 
          <TalkingHeadComponent/>
          <SimpleVoiceAssistant onConnectButtonClicked={onConnectButtonClicked} />
          </div>
        </div>
      </RoomContext.Provider>
    </div>
  );
}

function SimpleVoiceAssistant(props: { onConnectButtonClicked: () => void }) {
  const { state: agentState } = useVoiceAssistant();

  return (
    <>
      <AnimatePresence mode="wait">
        {agentState === "disconnected" ? (
          <motion.div
            key="disconnected"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.3, ease: [0.09, 1.04, 0.245, 1.055] }}
            className="grid items-center justify-center h-full flex-1"
          >
            <motion.button
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.3, delay: 0.1 }}
              className="uppercase px-4 py-2 bg-white text-black rounded-md flex-1"
              onClick={() => props.onConnectButtonClicked()}
            >
              Start a conversation
            </motion.button>
          </motion.div>
        ) : (
          <motion.div
            key="connected"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3, ease: [0.09, 1.04, 0.245, 1.055] }}
            className="flex flex-col items-center h-full m-20 flex-1"
          >
            <AgentVisualizer />
              <TranscriptionView onConnectButtonClicked={props.onConnectButtonClicked} />
            <RoomAudioRenderer volume={0.3}/>
            <NoAgentNotification state={agentState} />
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

function AgentVisualizer() {
  const { state: agentState, videoTrack, audioTrack } = useVoiceAssistant();

  if (videoTrack) {
    return (
      <div className="h-[512px] w-[512px] rounded-lg overflow-hidden">
        <VideoTrack trackRef={videoTrack} />
      </div>
    );
  }
  return (
    <div className="h-[300px] w-full">
      <BarVisualizer
        state={agentState}
        barCount={5}
        trackRef={audioTrack}
        className="agent-visualizer"
        options={{ minHeight: 24 }}
      />
    </div>
  );
}


function onDeviceFailure(error: Error) {
  console.error(error);
  alert(
    "Error acquiring camera or microphone permissions. Please make sure you grant the necessary permissions in your browser and reload the tab"
  );
}

import { useTrackTranscription, useVoiceAssistant } from "@livekit/components-react";
import { useCallback, useMemo, useState } from "react";
import useLocalMicTrack from "./useLocalMicTrack";
 interface TranscriptionSegment {
  id: string;
  text: string;
  language: string;
  startTime: number;
  endTime: number;
  final: boolean;
  firstReceivedTime: number;
  lastReceivedTime: number;
}
export type ReceivedTranscriptionSegment = TranscriptionSegment & {
  receivedAtMediaTimestamp: number;
  receivedAt: number;
};
export default function useCombinedTranscriptions() {
  const { agentTranscriptions } = useVoiceAssistant();

  const micTrackRef = useLocalMicTrack();
  const { segments: userTranscriptions } = useTrackTranscription(micTrackRef);
  const [manuallyAddedItems,setManuallyAddedItems]=useState<ReceivedTranscriptionSegment[]>([]);
  const addManual=useCallback((item:ReceivedTranscriptionSegment)=>{
    setManuallyAddedItems((val)=>{
      return [...val,item]
    })
  },[setManuallyAddedItems])
  const combinedTranscriptions = useMemo(() => {
    return [
      ...agentTranscriptions.map((val) => {
        return { ...val, role: "assistant" };
      }),
      ...userTranscriptions.map((val) => {
        return { ...val, role: "user" };
      }),
      ...manuallyAddedItems.map((val) => {
        return { ...val, role: "user" };
      }),
    ].sort((a, b) => a.firstReceivedTime - b.firstReceivedTime);
  }, [agentTranscriptions, userTranscriptions,manuallyAddedItems]);

  return {combinedTranscriptions,addManual};
}

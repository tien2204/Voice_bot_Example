import useCombinedTranscriptions from "@/hooks/useCombinedTranscriptions";
import {
  DisconnectButton,
  VoiceAssistantControlBar,
  useRoomContext,
  useVoiceAssistant,
} from "@livekit/components-react";
import { AnimatePresence, motion } from "framer-motion";
import * as React from "react";
import { text } from "stream/consumers";
import { CloseIcon } from "./CloseIcon";

export default function TranscriptionView(props: { onConnectButtonClicked: () => void }) {
  const { combinedTranscriptions, addManual } = useCombinedTranscriptions();
  const containerRef = React.useRef<HTMLDivElement>(null);

  // scroll to bottom when new transcription is added
  React.useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [combinedTranscriptions]);

  return (
    <>
      <div className="flex-grow w-full">
        <div className="relative h-[50vh] mx-auto">
          {/* Fade-out gradient mask */}
          <div className="absolute top-0 left-0 right-0 h-8 bg-gradient-to-b from-[var(--lk-bg)] to-transparent z-10 pointer-events-none" />
          <div className="absolute bottom-0 left-0 right-0 h-8 bg-gradient-to-t from-[var(--lk-bg)] to-transparent z-10 pointer-events-none" />

          {/* Scrollable content */}
          <div ref={containerRef} className="h-full flex flex-col gap-2 overflow-y-auto px-4 py-8">
            {combinedTranscriptions.map((segment) => (
              <div
                id={segment.id}
                key={segment.id}
                className={
                  segment.role === "assistant"
                    ? "p-2 self-start fit-content"
                    : "bg-gray-800 rounded-md p-2 self-end fit-content"
                }
              >
                {segment.text}
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="w-full">
        <ControlBar
          onConnectButtonClicked={props.onConnectButtonClicked}
          onTextSent={(text) => {
            const curtime = Date.now();
            addManual({
              id: "" + curtime,
              text: text,
              language: "en",
              startTime: curtime,
              endTime: curtime,
              final: true,
              firstReceivedTime: curtime,
              lastReceivedTime: curtime,
              receivedAtMediaTimestamp: curtime,
              receivedAt: curtime,
            });
          }}
        />
      </div>
    </>
  );
}

export function ControlBar(props: {
  onConnectButtonClicked: () => void;
  onTextSent?: (text: string) => void;
}) {
  const { state: agentState } = useVoiceAssistant();
  const room = useRoomContext();
  const [message, setMessage] = React.useState("");
  const submitText = () => {
    const cur_msg = message;
    console.log(cur_msg);
    if (cur_msg && typeof cur_msg === "string") {
      room.localParticipant.sendText(cur_msg, {
        topic: "custom-agent-text-input",
      });
    }
    props.onTextSent && props.onTextSent(cur_msg);
    setMessage("");
  };
  return (
    <div className="relative h-28 w-full">
      <AnimatePresence>
        {agentState === "disconnected" && (
          <motion.button
            initial={{ opacity: 0, top: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0, top: "-10px" }}
            transition={{ duration: 1, ease: [0.09, 1.04, 0.245, 1.055] }}
            className="uppercase absolute left-1/2 -translate-x-1/2 px-4 py-2 bg-white text-black rounded-md "
            onClick={() => props.onConnectButtonClicked()}
          >
            Start a conversation
          </motion.button>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {agentState !== "disconnected" && agentState !== "connecting" && (
          <motion.div
            initial={{ opacity: 0, top: "10px" }}
            animate={{ opacity: 1, top: "0" }}
            exit={{ opacity: 0, top: "-10px" }}
            transition={{ duration: 0.4, ease: [0.09, 1.04, 0.245, 1.055] }}
            className="flex flex-col h-full absolute left-1/2 -translate-x-1/2 justify-center gap-4 w-full"
          >
            <div style={{ display: "flex", flexDirection: "row" }}>
              <input
                type="text"
                placeholder="Enter text"
                className="flex-1 px-4 py-2 border rounded-lg focus:outline-none"
                onKeyDown={(evt) => {
                  if ((evt.key == "Enter" || evt.key === "NumpadEnter") && !evt.shiftKey) {
                    submitText();
                  }
                }}
                value={message}
                onChange={(e) => {
                  setMessage(e.target.value);
                }}
              />
              <button
                className="uppercase ml-2 px-4 py-2 bg-white text-black rounded-md"
                onClick={submitText}
              >
                Send
              </button>
            </div>
            <div className="flex" style={{ justifyContent: "end" }}>
              <VoiceAssistantControlBar controls={{ leave: false }} />
              <DisconnectButton>
                <CloseIcon />
              </DisconnectButton>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

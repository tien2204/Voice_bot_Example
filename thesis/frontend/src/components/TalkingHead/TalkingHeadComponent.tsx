import {
  AgentState,
  DisconnectButton,
  useTrackTranscription,
  useVoiceAssistant,
} from "@livekit/components-react";

import type { Participant, Track, TrackPublication, TranscriptionSegment } from "livekit-client";
import React, { useEffect, useMemo, useRef } from "react";
import { Audio, TalkingHead } from "./talkinghead.mjs";
import { useAuth } from "../../contexts/AuthContext"; // Import useAuth
import { getLookAtFromOverlaps, YoloWebcam } from "./YoloWebcam";
import { useTalkingHeadStore } from "../../store/talkingHeadStore";
import { CloseIcon } from "@/components/CloseIcon";
import {Avatar} from "@/types/avatar"
import { BasicHeadDisplay } from "./BasicHeadDisplay";
// ## TrackReference Types

/** @public */
export type TrackReferencePlaceholder = {
  participant: Participant;
  publication?: never;
  source: Track.Source;
};

/** @public */
export type TrackReference = {
  participant: Participant;
  publication: TrackPublication;
  source: Track.Source;
};

/** @public */
export type TrackReferenceOrPlaceholder = TrackReference | TrackReferencePlaceholder;

interface AudioInput {
  status: number;
  audio: Audio;
  opt: object;
}

interface TalkingHeadAppProps {
  state?: AgentState;
  /** @deprecated The trackRef is now automatically derived from `useVoiceAssistant` */
  // trackRef is no longer used as a prop, derived from useVoiceAssistant
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  trackRef?: TrackReferenceOrPlaceholder;
}
function createSilentArrayBuffer(
  duration: number,
) {
  const length = Math.max(Math.floor(duration), 10);
  const buffer = new ArrayBuffer(length);
  return buffer;
}


function convertTranscriptionEventToAudio(
  processedSegment: Pick<TranscriptionSegment, "text"> // We only need the text of the diff_chunk
): Audio {
  const words = processedSegment.text.trim().split(/\s+/).filter(w => w.length > 0);

  if (words.length === 0) {
    return {
      audio: createSilentArrayBuffer(10),
      words: [],
      wtimes: [],
      wdurations: [],
    };
  }

  const wordCount = words.length;

  // Estimate duration for the diff_chunk.
  const estimatedMsPerWord = 300; // Average duration per word in ms (e.g., ~200 words/min)
  const totalDurationMs = words.length * estimatedMsPerWord;

  // wtimes and wdurations should be relative to the start of this diff_chunk's audio, in SECONDS.
  const wtimes = words.map((_, i) => (i * estimatedMsPerWord));
  const wdurations = words.map(() => estimatedMsPerWord);
  const totalDuration = totalDurationMs;

  const audioBuffer = createSilentArrayBuffer(totalDuration); // Use 48000Hz

  const audio: Audio = {
    audio: audioBuffer,
    words,
    wtimes: wtimes,
    wdurations: wdurations,
  };
  console.log(audio)
  return audio;
}

export function TalkingHeadComponent({}: TalkingHeadAppProps) {
  const { state, videoTrack, audioTrack: trackReference } = useVoiceAssistant();
  const { user, isAuthenticated, token:authToken } = useAuth(); // Get user, auth status, and token from AuthContext
  const subtitlesRef = useRef<HTMLDivElement | null>(null);

  const {
    subtitle,
    info,
    err,
    head,
    allModelUrls,
    currentModelIndex,
    modelUrl,
    coordinates,
    // Actions
    appendSubtitle,
    setCoordinates,
    fetchUserAvatars,
    loadAvatarFromModelUrl,
    selectNextModel,
    selectPreviousModel,
  } = useTalkingHeadStore();

  async function streamStartIfNot(...args:Parameters<TalkingHead["streamStart"]>){
    if(!args[0]||!("lipsyncType" in args[0])){
      args[0]={...args[0]??{},lipsyncType:"words"}
    }
    if(!head){
      return
    }
    if(!head.isStreaming){
      await head.streamStart(
        ...args
      )
    } // Missing closing brace was a typo in my interpretation, original code is fine here.
    return // Original code is fine here.
  }

  useEffect(() => {
    if (subtitlesRef.current) {
      subtitlesRef.current.scrollTop = subtitlesRef.current.scrollHeight;
    }
  }, [subtitle]);

  const preAudio = useRef<Audio | undefined>();
  useTrackTranscription(trackReference, {
    onTranscription: (events) => {
      streamStartIfNot(
        {
          lipsyncType: "words",
        },
        null,
        null,
        // @ts-ignore
        (word: Object) => {
          appendSubtitle(word.toString());
        }
      ).then(() => {
        useTalkingHeadStore.setState((state) => {
          const newTranscriptions = { ...state.transcriptions }; // Work with a copy
          for (const ev of events) {
            let diff_chunk = ev.text;
            // Use the newTranscriptions copy for reading and writing
            if (newTranscriptions[ev.id] && newTranscriptions[ev.id].text) {
              diff_chunk = ev.text.slice(newTranscriptions[ev.id].text.length);
            }
            newTranscriptions[ev.id] = ev; // Update the copy

            diff_chunk = diff_chunk.trim();
            if (!diff_chunk) {
              continue;
            }
            let produced_audio=convertTranscriptionEventToAudio({...ev, text: diff_chunk });
            let fixed=(preAudio.current&&preAudio.current.wtimes)?preAudio.current.wtimes[preAudio.current.wtimes.length-1]:0
            produced_audio.wtimes=produced_audio.wtimes.map(
              (time,id)=>time+fixed+produced_audio.wdurations[id]
            )
            head?.streamAudio(produced_audio);
            preAudio.current = produced_audio;
          }
          return { transcriptions: newTranscriptions }; // Return the updated copy
        });
      });
    }
  });

  useEffect(() => {
    if (isAuthenticated && authToken) {
      fetchUserAvatars(authToken);
    } else {
      // Clear user models if not authenticated
      useTalkingHeadStore.getState().setUserModels([]);
    }
  }, [isAuthenticated, user, authToken]);

  useEffect(() => {
    // This effect is to trigger re-calculation of modelUrl if allModelUrls/currentModelIndex changes from store internals
  }, [allModelUrls, currentModelIndex, modelUrl]);

  useEffect(()=>{
    let intervalId:ReturnType<typeof setInterval>|undefined=undefined;

    // This effect periodically makes the avatar look at the current `coordinates`.
    // It reads `coordinates` from the store (via the hook) but does not set them.
    // `coordinates` are set by YoloWebcam.
    if(info === 1 && head && head.avatar){
      intervalId = setInterval(()=>{
        // `coordinates` is from the useTalkingHeadStore() hook call
        const currentCoords = coordinates; 

        if(!currentCoords){
          // head and head.avatar are confirmed by the outer if condition
          head.lookAtCamera(1,320);
        } else {
          try{
            // @ts-ignore
            const rect = head.nodeAvatar.getBoundingClientRect();
            head.lookAt(
              (currentCoords[0]-1/2)*rect.width+rect.left,
              (currentCoords[1]-1/2)*rect.height+rect.top,
              1,320
            );
          } catch(e){
            console.error("Error in lookAt:", e);
            head.lookAtCamera(1,320); // Fallback
          }
        }
      }, 100); // Increased interval to 100ms (adjust as needed)
    }
    return ()=>{
      if(intervalId) clearInterval(intervalId);
    }
  },[info, head, coordinates]); // Dependencies: info, head, and current coordinates

  useEffect(()=>{
    try{
      if(info === 1 && state && head){
        if (state === "disconnected") {
          // head?.lookAhead(1)
          head?.setMood("neutral");
          head?.playPose("wide");

        } else {
          head?.playPose("straight");
          head?.speakEmoji("👋");
          head?.setMood("happy");
        }
      }
    } catch(e){
      console.error(e)
    }

  },[state, head, info]) // state === "disconnected" was too specific, listen to broader changes
  useEffect(() => {
    if (info === 1 && state && head) {
      try {

        // if(state==="listening"){
        //   // @ts-ignore
        //   head?.startListening(AnalyserNode(),{})
        // }
        // if (state !== "speaking") {
        //   head?.stopSpeaking();
        // }
        if (state === "speaking") {
            
        } else if(state=="listening"){
            preAudio.current=undefined;
            head?.stopSpeaking();
        } else {
          // head?.streamStop()
          // head?.streamStart(
          //   {},null,null,
          //   // @ts-ignore
          //   (word: Object) => {
          //     addSubtitle(word.toString());
          //   }
          // )
        }
      } catch (e) {
        console.error(e);
      }
    }
  }, [state, head]);

  const speakRandom = () => {
    streamStartIfNot(
      {
        lipsyncType:"words"
      },
      null,
      null,
      // @ts-ignore
      (word: Object) => {
        appendSubtitle(word.toString());
      }
    ).then(()=>{
      head?.streamAudio(
        {
          words: ["Yes",",","Life ", "is ", "like ", "a ", "box "],
          wtimes: [493, 1009, 1116, 1269, 1349, 1677, 1779],
          wdurations: [459, 91, 138, 39, 313, 63, 959],
          audio: new ArrayBuffer()
          // @ts-ignore
          // audio: new (window.AudioContext || window.webkitAudioContext)().createBuffer(
          //   1,
          //   142800,
          //   48000
          // ),
        },
        // {},
        // (word: Object) => {
        //   console.log("Caption")
        //   appendSubtitle(word.toString());
        // }
      );
      head?.startSpeaking()
      // head?.streamNotifyEnd();
    })
  };

  const basicHeadInitOptions = useMemo(() => ({
    ttsEndpoint: "N/A", // This should be configurable based on your app's needs
    cameraView: "full", // Default for the main component
    mixerGainSpeech: 3,
    // Add any other options that are static or change infrequently
    // and ensure their dependencies are in useMemo's array if they can change.
    // For now, assuming these are static for the component's lifetime.
  }), []);

  return (
    <div style={styles.container}>
      <BasicHeadDisplay
        style={styles.avatar}
        initOptions={basicHeadInitOptions}
      />
      {/* {info !== 1 && !err && <div style={styles.info}>Loading {Math.round(info * 100)}%...</div>} */}
      {err && <div style={styles.info}>{err}</div>}
      <div ref={subtitlesRef} style={styles.subtitles}>
        {subtitle}
      </div>
      <div style={{...styles.modelSwitcherControls,
        ...((allModelUrls.length <= 1 || state !== "disconnected")?{display:"none"}:{})
      }}>
        <button onClick={selectPreviousModel} style={styles.modelButton} disabled={allModelUrls.length <= 1 || state !== "disconnected"}>&lt;</button>
        <span style={styles.modelIndicator}>{allModelUrls.length > 0 ? currentModelIndex + 1 : 0} / {allModelUrls.length}</span>
        <button onClick={selectNextModel} style={styles.modelButton} disabled={allModelUrls.length <= 1 || state !== "disconnected"}>&gt;</button>
      </div>
      <YoloWebcam onOverlapChange={(overlaps)=>{
        setCoordinates((oldxy)=>{
          const xy=getLookAtFromOverlaps(overlaps)
          if(!xy){
            return xy;
          }
          if(oldxy&&xy[0]==oldxy[0]&&xy[1]==oldxy[1]){
            return oldxy;
          }
          return xy;
        })
      }}/>
      {/* <button style={{ width: "10vw", height: "10vh", background: "red" }} onClick={speakRandom}>
        Speak Random
      </button>
      <button
        style={{ width: "10vw", height: "10vh", background: "red" }}
        onClick={() => {
          head?.speakEmoji("👋");
        }}
      >
        Gesture
      </button> */}
    </div>
  );
}

const styles: { [key: string]: React.CSSProperties } = {
  container: {
    flex:1,
    height: "100%",
    // margin: "auto",
    // position: "relative",
    // backgroundColor: "#202020",
    color: "white",
    // flex:1
  },
  avatar: {
    display: "block",
    width: "100%",
    height: "100%",
  },
  controls: {
    position: "absolute",
    top: 10,
    left: 10,
    right: 10,
    height: 40,
  },
  textInput: {
    position: "absolute",
    width: "calc(100% - 140px)",
    height: 40,
    top: 0,
    left: 0,
    padding: "0 10px",
    fontFamily: "Arial",
    fontSize: 20,
  },
  button: {
    position: "absolute",
    right: 10,
    height: 40,
    width: 85,
    fontFamily: "Arial",
    fontSize: 20,
  },
  info: {
    position: "absolute",
    bottom: 10,
    left: 10,
    right: 10,
    height: 50,
    fontFamily: "Arial",
    fontSize: 20,
  },
  subtitles: {
    position: "absolute",
    bottom: "6vh",
    left: "50%",
    transform: "translateX(-50%)",
    fontFamily: "Arial",
    fontSize: "max(min(5vh, 5vw), 24px)",
    lineHeight: "max(min(6vh, 6vh), 20px)",
    zIndex: 30,
    height: "calc(2 * max(min(6vh, 6vh), 20px))",
    maxHeight: "calc(2 * max(min(6vh, 6vh), 20px))",
    width: "80%",
    textAlign: "center",
    overflow: "hidden",
  },
  modelSwitcherControls: {
    position: "absolute",
    bottom: "10px",
    left: "50%",
    transform: "translateX(-50%)",
    zIndex: 40,
    display: "flex",
    alignItems: "center",
    gap: "10px",
    backgroundColor: "rgba(0,0,0,0.6)",
    padding: "8px 12px",
    borderRadius: "8px",
  },
  modelButton: {
    background: "#555",
    color: "white",
    border: "1px solid #777",
    padding: "8px 15px",
    cursor: "pointer",
    borderRadius: "5px",
    fontSize: "16px",
    lineHeight: "1",
  },
  disconnectButton: {
    position: "absolute",
    top: "20px",
    right: "20px",
    zIndex: 50,
    background: "rgba(0,0,0,0.5)",
    borderRadius: "50%",
  }
};

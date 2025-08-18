import { Avatar } from "@/types/avatar";
import type { TranscriptionSegment } from "livekit-client";
import { create } from "zustand";
// Adjust the import paths based on your project structure
import type { Audio, TalkingHead } from "../components/TalkingHead/talkinghead.mjs";

const DEFAULT_MODEL_URLS = [
  "https://models.readyplayer.me/64bfa15f0e72c63d7c3934a6.glb?morphTargets=ARKit,Oculus+Visemes,mouthOpen,mouthSmile,eyesClosed,eyesLookUp,eyesLookDown&textureSizeLimit=1024&textureFormat=webp",
];

const AVATAR_QUERY_PARAMS_STRING = "morphTargets=ARKit,Oculus+Visemes,mouthOpen,mouthSmile,eyesClosed,eyesLookUp,eyesLookDown&textureSizeLimit=1024&textureFormat=webp";

interface TalkingHeadState {
  subtitle: string;
  info: number; // loading progress 0-1
  err?: string;
  head?: TalkingHead;

  userModels: string[]; // URLs from user's saved avatars
  allModelUrls: string[]; // Combined userModels + DEFAULT_MODEL_URLS
  currentModelIndex: number;
  modelUrl: string; // Current active model URL to load

  coordinates?: number[]; // For YoloWebcam lookAt
  transcriptions: { [id: string]: TranscriptionSegment };

  // Actions
  setSubtitle: (subtitle: string) => void;
  appendSubtitle: (word: string) => void;
  forceClearSubtitle: () => void;

  setInfo: (info: number) => void;
  setErr: (err?: string) => void;
  setHead: (head?: TalkingHead) => void;

  _updateAllModelUrls: () => void;
  setUserModels: (models: string[]) => void;
  setCurrentModelIndex: (index: number) => void;
  selectNextModel: () => void;
  selectPreviousModel: () => void;

  setCoordinates: (f: (coords?: number[]) => number[] | undefined) => void;

  addOrUpdateTranscription: (id: string, segment: TranscriptionSegment) => void;
  clearTranscriptions: () => void;

  fetchUserAvatars: (authToken: string) => Promise<void>;
  fetchRpmAuthToken: (authToken: string) => Promise<string|undefined>;

  initializeHeadInstance: (
    container: HTMLDivElement,
    options: ConstructorParameters<typeof TalkingHead>[1]
  ) => Promise<void>;
  loadAvatarFromModelUrl: () => Promise<void>;
  destroyHeadInstance: () => void;
}

let subtitleClearTimer: NodeJS.Timeout | undefined;

const addAvatarParamsToUrl = (urlString: string): string => {
  if (!urlString || typeof urlString !== 'string') {
    // Return empty or original string if input is not a valid string for URL processing
    return urlString || "";
  }
  try {
    const url = new URL(urlString); // Assumes urlString is an absolute URL
    const paramsToApply = new URLSearchParams(AVATAR_QUERY_PARAMS_STRING);
    paramsToApply.forEach((value, key) => {
      url.searchParams.set(key, value); // Add or overwrite existing parameters
    });
    return url.toString();
  } catch (error) {
    console.warn(`Could not parse or modify URL "${urlString}" to add avatar params. Returning original. Error: ${error}`);
    return urlString; // Return original URL if parsing/modification fails
  }
};

export const useTalkingHeadStore = create<TalkingHeadState>((set, get) => ({
  subtitle: "",
  info: 0,
  err: undefined,
  head: undefined,

  userModels: [],
  allModelUrls: [...DEFAULT_MODEL_URLS],
  currentModelIndex: 0,
  modelUrl: DEFAULT_MODEL_URLS[0] || "",

  rpmAuthToken: null,
  coordinates: undefined,
  transcriptions: {},

  setSubtitle: (subtitle) => set({ subtitle }),
  appendSubtitle: (word) => {
    if (subtitleClearTimer) clearTimeout(subtitleClearTimer);
    set((state) => ({ subtitle: (state.subtitle + " " + word).trim() }));
    subtitleClearTimer = setTimeout(() => {
      set({ subtitle: "" });
      subtitleClearTimer = undefined;
    }, 3000); // Auto-clear delay
  },
  forceClearSubtitle: () => {
    if (subtitleClearTimer) {
      clearTimeout(subtitleClearTimer);
      subtitleClearTimer = undefined;
    }
    set({ subtitle: "" });
  },

  setInfo: (info) => set({ info }),
  setErr: (err) => set({ err }),
  setHead: (head) => set({ head }),

  _updateAllModelUrls: () => {
    set((state) => {
      const combined = [
        ...state.userModels,
        ...DEFAULT_MODEL_URLS.filter((url) => !state.userModels.includes(url)),
      ];
      console.log(combined)
      const newAllModelUrls = combined.length > 0 ? combined : [...DEFAULT_MODEL_URLS];
      let newCurrentModelIndex = state.currentModelIndex;
      // Ensure index is valid, defaulting to 0 if list becomes empty or shorter
      if (newAllModelUrls.length === 0) {
        newCurrentModelIndex = 0;
      } else if (newCurrentModelIndex >= newAllModelUrls.length) {
        newCurrentModelIndex = Math.max(0, newAllModelUrls.length - 1);
      }
      const newModelUrl = newAllModelUrls[newCurrentModelIndex] ;
      return {
        allModelUrls: newAllModelUrls,
        currentModelIndex: newCurrentModelIndex,
        modelUrl: newModelUrl,
      };
    });
  },
  setUserModels: (models) => {
    const processedModels = models.map(modelUrl => addAvatarParamsToUrl(modelUrl));
    set({ userModels: processedModels });
    get()._updateAllModelUrls();
  },
  setCurrentModelIndex: (index) => {
    set((state) => {
      if (state.allModelUrls.length === 0) return { currentModelIndex: 0, modelUrl: "" };
      const newIndex = Math.max(0, Math.min(index, state.allModelUrls.length - 1));
      return {
        currentModelIndex: newIndex,
        modelUrl: state.allModelUrls[newIndex] || "",
      };
    });
  },
  selectNextModel: () => {
    get().setCurrentModelIndex((get().currentModelIndex + 1) % (get().allModelUrls.length || 1));
  },
  selectPreviousModel: () => {
    get().setCurrentModelIndex(
      (get().currentModelIndex - 1 + (get().allModelUrls.length || 1)) %
        (get().allModelUrls.length || 1)
    );
  },

  setCoordinates: (f) => set(({ coordinates }) => ({ coordinates: f(coordinates) })),

  addOrUpdateTranscription: (id, segment) => {
    set((state) => ({
      transcriptions: { ...state.transcriptions, [id]: segment },
    }));
  },
  clearTranscriptions: () => set({ transcriptions: {} }),

  fetchUserAvatars: async ( authToken) => {
    if ( !authToken) {
      get().setUserModels([]);
      return;
    }
    try {
      const response = await fetch(process.env.NEXT_PUBLIC_BACKEND_URL + `/users/me/avatars`, {
        headers: { Authorization: `Bearer ${authToken}` },
      });
      if (response.ok) {
        const avatars: Avatar[] = await response.json();
        get().setUserModels(avatars.map((avatar) => avatar.glb_url).filter((val)=>val));
      } else {
        console.error("Failed to fetch user avatars:", await response.text());
        get().setUserModels([]);
      }
    } catch (error) {
      console.error("Error fetching user avatars:", error);
      get().setUserModels([]);
    }
  },

  fetchRpmAuthToken: async (authToken) => {
    if (!authToken) {
      return;
    }
    try {
      const response = await fetch(process.env.NEXT_PUBLIC_BACKEND_URL + `/users/me/rpm-auth-token`, {
        headers: { 'Authorization': `Bearer ${authToken}` },
      });
      if (response.ok) {
        const resp = await response.json();
        return resp.token
      } else {
        console.error('Failed to fetch RPM auth token:', await response.text());
      }
    } catch (error) {
      console.error('Error fetching RPM auth token:', error);
    }
    return undefined;
  },

  initializeHeadInstance: async (container, options) => {
    if (get().head) get().destroyHeadInstance();
    if (container.children.length > 0) container.replaceChildren();
    const { TalkingHead } = await import("@/components/TalkingHead/talkinghead.mjs");
    container.textContent = "";
    const instance = new TalkingHead(container, options);
    set({ head: instance, info: 0, err: undefined });
  },

  loadAvatarFromModelUrl: async () => {
    const { head, modelUrl, setInfo, setErr } = get();
    if (!head || !modelUrl) {
      setInfo(0);
      if (!modelUrl && head) setErr("No model URL specified.");
      else if (!head) setErr("TalkingHead not initialized.");
      return;
    }
    try {
      setInfo(0);
      setErr(undefined);
      await head.showAvatar(
        { url: modelUrl, body: "F", avatarMood: "happy", lipsyncLang: "en" },
        (ev) => {
          if (ev.lengthComputable) setInfo(Math.min(1, ev.loaded / ev.total));
        }
      );
      setInfo(1);
    } catch (error: any) {
      setErr(error.toString());
      setInfo(0);
    }
  },

  destroyHeadInstance: () => {
    const { head } = get();
    if (head) {
      try {
        head.stop(); // Stops animation loop and suspends audio context
        try {
          head.stopSpeaking();
        } catch (e) {}
        try {
          head.streamStop();
        } catch (e) {}
        if (head.gestureTimeout) {
          clearTimeout(head.gestureTimeout);
          head.gestureTimeout = null;
        }

        if (head.renderer) {
          head.renderer.dispose();
        }
        if (head.audioCtx && head.audioCtx.state !== "closed") {
          head.audioCtx.close();
        }
        if (head.resizeobserver) {
          head.resizeobserver.disconnect();
        }
        head.dynamicbones.dispose();
        try {
          // @ts-expect-error
          head.nodeAvatar.removeChild(head.renderer.domElement);
        } catch (e) {}
      } catch (e) {
        console.error(e);
      }
    }
    set({ head: undefined, info: 0 });
  },
}));

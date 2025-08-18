"use client";

import { useAuth } from "@/contexts/AuthContext";
import { useTalkingHeadStore } from "@/store/talkingHeadStore";
import { useRouter } from "next/navigation";
import React, { useEffect, useRef, useState } from "react";

const RPM_SUBDOMAIN = process.env.NEXT_PUBLIC_RPM_SUBDOMAIN;
interface AvatarCreatorComponentProps {
  onAvatarExported?: (avatarUrl: string) => void;
}

const styles: { [key: string]: React.CSSProperties } = {
  container: {
    width: "100%",
    height: "100vh", // Adjust height as needed
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#f0f0f0",
  },
  iframe: {
    width: "100%",
    height: "100%",
    border: "none",
    // borderRadius: '8px',
    boxShadow: "0 4px 8px rgba(0,0,0,0.1)",
  },
  infoText: {
    marginTop: "20px",
    fontSize: "16px",
    color: "#333",
  },
  button: {
    marginTop: "10px",
    padding: "10px 20px",
    fontSize: "16px",
    color: "white",
    backgroundColor: "#007bff",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
  },
};

export function AvatarCreatorComponent({ onAvatarExported }: AvatarCreatorComponentProps) {
  const router = useRouter();
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const [avatarUrl, setAvatarUrl] = useState<string | null>(null);
  const [isCreatorReady, setIsCreatorReady] = useState(false);
  const [isRpmTokenLoading, setIsRpmTokenLoading] = useState(false);
  const [rpmAuthToken,setRpmAuthToken] = useState<string|undefined>();
  const { user, isAuthenticated, token: authToken, fetchCurrentUser } = useAuth(); // Get user, auth status, and authToken from AuthContext
  const { fetchRpmAuthToken } = useTalkingHeadStore();

  // Definition for the payload to be sent to the backend
  interface AvatarPayload {
    glb_url: string;
    rpm_avatar_id?: string;
    config?: { [key: string]: any };
  }

  useEffect(() => {
    if (isAuthenticated && authToken && !rpmAuthToken) {
      setIsRpmTokenLoading(true);
      fetchRpmAuthToken(authToken).then((val) => {
        setRpmAuthToken(val)
      }).finally(()=>{
        setIsRpmTokenLoading(false);
      });
    }
  }, [isAuthenticated, authToken, rpmAuthToken, fetchRpmAuthToken]); // No need to add setIsRpmTokenLoading to deps

  useEffect(() => {
    const handleMessage = async (event: MessageEvent) => {
      try {
        const json = typeof event.data === "string" ? JSON.parse(event.data) : event.data;

        if (json?.source !== "readyplayerme") {
          return;
        }

        // Subscribe to all events sent from Ready Player Me
        // See https://docs.readyplayer.me/ready-player-me/iframe-api/events
        console.log(json.eventName)
        if (json.eventName === "v1.frame.ready") {
          setIsCreatorReady(true);
          if (iframeRef.current?.contentWindow) {
            iframeRef.current.contentWindow.postMessage(
              JSON.stringify({
                target: "readyplayerme",
                type: "subscribe",
                eventName: "v1.**",
              }),
              "*"
            );
          }
        }

        if (json.eventName === "v1.avatar.exported") {
          const exportedData = json.data as { url: string; avatarId?: string; [key: string]: any };
          const newAvatarGlbUrl = exportedData.url;

          setAvatarUrl(newAvatarGlbUrl); // Update UI state
          console.log(
            `Avatar exported: ${newAvatarGlbUrl}, RPM Avatar ID: ${exportedData.avatarId}`
          );

          if (onAvatarExported) {
            onAvatarExported(newAvatarGlbUrl); // Notify parent component
          }

          // Prepare avatar data for saving, matching backend's AvatarMetadata model
          const avatarPayload: AvatarPayload = {
            glb_url: newAvatarGlbUrl,
            rpm_avatar_id: exportedData.avatarId, // Store the Ready Player Me specific avatar ID
            config: {
              source: "readyplayerme",
            },
          };

          if (isAuthenticated && authToken) {
            await saveAvatarToBackend(avatarPayload, authToken).then(() => {
              router.push("/avatar/chat");
            });
            fetchCurrentUser?.(); // Refresh user data to get updated avatars list
          } else {
            console.warn("User not authenticated or no token found. Avatar not saved to backend.");
          }
        }
      } catch (error) {
        console.error("Error processing message from iframe:", error);
      }
    };

    window.addEventListener("message", handleMessage);
    return () => window.removeEventListener("message", handleMessage);
  }, [onAvatarExported, isAuthenticated, authToken, fetchCurrentUser, router]); // user removed as it might cause stale closures if not careful

  const saveAvatarToBackend = async (avatarPayload: AvatarPayload, token: string) => {
    console.log(`Saving new avatar to backend:`, avatarPayload);
    try {
      // Endpoint changed to /me/avatars, user identified by token
      const response = await fetch(process.env.NEXT_PUBLIC_BACKEND_URL + `/users/me/avatars`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(avatarPayload),
      });
      if (response.ok) {
        const updatedUser = await response.json();
        console.log("Avatar added successfully. Updated user:", updatedUser);
      } else {
        console.error("Failed to save avatar to backend:", await response.text());
      }
    } catch (error) {
      console.error("Error saving avatar to backend:", error);
    }
  };

  // The user.rpm_guest_user_id is available here if needed for other purposes,
  // but not directly used in the iframe src for standard guest avatar creation.
  // console.log("Current user's RPM Guest User ID:", user?.rpm_guest_user_id);

  return (
    <div style={styles.container}>
      {isRpmTokenLoading ? (
        <p style={styles.infoText}>Fetching RPM token...</p>
      ) : !rpmAuthToken ? (
        <p style={{ ...styles.infoText, color: "red" }}>
          Failed to fetch RPM token. Please ensure you are logged in and try again.
        </p>
      ) : (
        <iframe
          key={rpmAuthToken} // Add key to re-render iframe when token changes
          ref={iframeRef}
          src={
            rpmAuthToken
              ? `https://${RPM_SUBDOMAIN}.readyplayer.me/avatar?frameApi&token=${rpmAuthToken}`
              : undefined
          }
          style={styles.iframe}
          allow="camera *; microphone *; clipboard-write"
          title="Ready Player Me Avatar Creator"
        ></iframe>
      )}
      {/* {avatarUrl && <p style={styles.infoText}>Avatar URL: {avatarUrl}</p>} */}
    </div>
  );
}

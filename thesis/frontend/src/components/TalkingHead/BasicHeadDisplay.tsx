"use client";

import React, { useEffect, useRef } from 'react';
import { useTalkingHeadStore } from '../../store/talkingHeadStore';
import type { TalkingHead } from './talkinghead.mjs'; // Assuming TalkingHead class is accessible

const DEFAULT_FALLBACK_MODEL_URL = "/models/rpm_idle_test_1.vrm"; // Replace with your actual default model URL

interface BasicHeadDisplayProps {
  modelUrl?: string; // Specific model to load, otherwise store's current
  initialMood?: string;
  initialPose?: string;
  lookAtCamera?: boolean;
  onLoaded?: () => void;
  className?: string;
  style?: React.CSSProperties;
  initOptions?: any;
}

export function BasicHeadDisplay({
  modelUrl: propModelUrl,
  initialMood = "neutral",
  initialPose = "straight",
  lookAtCamera = false,
  onLoaded,
  className,
  style,
  initOptions,
}: BasicHeadDisplayProps) {
  const avatarRef = useRef<HTMLDivElement | null>(null);
  const {
    head,
    info,
    err,
    modelUrl: storeModelUrl,
    initializeHeadInstance,
    loadAvatarFromModelUrl,
    destroyHeadInstance,
  } = useTalkingHeadStore();

  const effectiveModelUrl = propModelUrl || storeModelUrl || DEFAULT_FALLBACK_MODEL_URL;

  useEffect(() => {
    if (!avatarRef.current) return;
    const currentAvatarRef = avatarRef.current;

    initializeHeadInstance(currentAvatarRef, {
      cameraView: "head", 
      ttsEndpoint: "N/A",
      ...initOptions??{},
    });

    const handleVisibilityChange = () => {
      const currentHead = useTalkingHeadStore.getState().head;
      if (document.visibilityState === 'visible') currentHead?.start();
      else currentHead?.stop();
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      destroyHeadInstance();
    };
  }, [initializeHeadInstance, destroyHeadInstance, initOptions]);

  useEffect(() => {
    if (head && effectiveModelUrl) {
      let isMounted = true;
      if (useTalkingHeadStore.getState().modelUrl !== effectiveModelUrl || !useTalkingHeadStore.getState().allModelUrls.includes(effectiveModelUrl)) {
        useTalkingHeadStore.setState({ modelUrl: effectiveModelUrl, allModelUrls: [effectiveModelUrl], currentModelIndex: 0 });
      }
      
      loadAvatarFromModelUrl().then(() => {
        if (isMounted && useTalkingHeadStore.getState().info === 1) {
          if (initialMood) head?.setMood(initialMood);
          if (initialPose) head?.playPose(initialPose);
          if (onLoaded) onLoaded();
        }
      });
      return () => { isMounted = false; };
    }
  }, [head, effectiveModelUrl, loadAvatarFromModelUrl, initialMood, initialPose, onLoaded, head?.setMood, head?.playPose]);

  useEffect(() => {
    let intervalId: NodeJS.Timeout | undefined;
    if (lookAtCamera && head?.avatar && info === 1) {
      head.lookAtCamera(1, 320); 
      intervalId = setInterval(() => {
        if (document.visibilityState === 'visible' && head?.avatar) head.lookAtCamera(1, 320);
      }, 5000);
    }
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [lookAtCamera, head, info]);
  
  const showLoadingText = info !== 1 && !err;
  const showErrorText = !!err;

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative', backgroundColor: initOptions?.modelBgColor || 'transparent', ...style }} className={className}>
      <div ref={avatarRef} style={{ width: '100%', height: '100%' }}></div>
      {showLoadingText && (
        <div style={{
          position: 'absolute',
          bottom: 10,
          left: '50%',
          transform: 'translateX(-50%)',
          textAlign: 'center',
          color: 'white',
          backgroundColor: 'rgba(0, 0, 0, 0.5)',
          padding: '2px 8px',
          borderRadius: '4px',
          fontSize: '1.5em'
        }}>Loading Model {Math.round(info * 100)}%...</div>
      )}
      {showErrorText && (
        <div style={{
          position: 'absolute',
          bottom: 10,
          left: '50%',
          transform: 'translateX(-50%)',
          textAlign: 'center',
          color: 'white',
          backgroundColor: 'rgba(200, 0, 0, 0.7)',
          padding: '2px 8px',
          borderRadius: '4px',
          fontSize: '1.5em'
        }}>Error: {err}</div>
      )}
    </div>
  );
}
import useSize from "@react-hook/size";
import * as ort from "onnxruntime-web";
import React, { PropsWithChildren, useEffect, useRef, useState } from "react";

type Detection = {
  score: number;
  box: [number, number, number, number];
};
// Box format: [x1, y1, x2, y2]
type Box = [number, number, number, number];

// Compute area of boxes
function areaOf(leftTop: number[], rightBottom: number[]): number {
  const hw = [Math.max(rightBottom[0] - leftTop[0], 0), Math.max(rightBottom[1] - leftTop[1], 0)];
  return hw[0] * hw[1];
}

// IoU between two boxes
function iouOf(box0: Box, box1: Box, eps = 1e-5): number {
  const overlapLeftTop = [Math.max(box0[0], box1[0]), Math.max(box0[1], box1[1])];
  const overlapRightBottom = [Math.min(box0[2], box1[2]), Math.min(box0[3], box1[3])];
  const overlapArea = areaOf(overlapLeftTop, overlapRightBottom);
  const area0 = areaOf([box0[0], box0[1]], [box0[2], box0[3]]);
  const area1 = areaOf([box1[0], box1[1]], [box1[2], box1[3]]);
  return overlapArea / (area0 + area1 - overlapArea + eps);
}

// Hard NMS
function hardNMS(
  boxesScores: Array<[number, number, number, number, number]>,
  iouThreshold: number,
  topK = -1,
  candidateSize = 200
): Array<[number, number, number, number, number]> {
  const scores = boxesScores.map((b) => b[4]);
  // Sort indexes by score ascending
  const indexes = scores
    .map((score, i) => i)
    .sort((a, b) => scores[a] - scores[b])
    .slice(-candidateSize);

  const picked: number[] = [];
  while (indexes.length > 0) {
    const current = indexes[indexes.length - 1];
    picked.push(current);
    if (topK > 0 && picked.length >= topK) break;
    indexes.pop();

    const currentBox = boxesScores[current].slice(0, 4) as Box;
    const restIndexes = indexes.slice();
    for (let i = restIndexes.length - 1; i >= 0; i--) {
      const box = boxesScores[restIndexes[i]].slice(0, 4) as Box;
      const iou = iouOf(box, currentBox);
      if (iou > iouThreshold) {
        indexes.splice(i, 1); // remove from indexes
      }
    }
  }
  return picked.map((i) => boxesScores[i]);
}

function predict(
  width: number,
  height: number,
  confidences: Float32Array, // shape: (1, N, 2)
  boxes: Float32Array, // shape: (1, N, 4)
  probThreshold: number,
  iouThreshold = 0.5,
  topK = -1
): { boxes: Box[]; labels: number[]; probs: number[] } {
  const pickedBoxProbs: Array<[number, number, number, number, number]> = [];
  const pickedLabels: number[] = [];

  const numBoxes = boxes.length / 4; // since shape is (1, N, 4)
  const numClasses = 2;

  for (let classIndex = 1; classIndex < numClasses; classIndex++) {
    const probs: number[] = [];

    // Extract class probs for classIndex
    for (let i = 0; i < numBoxes; i++) {
      const p = confidences[i * numClasses + classIndex]; // (1,N,2) flattened
      probs.push(p);
    }

    const mask = probs.map((p) => p > probThreshold);
    const filteredProbs = probs.filter((_, i) => mask[i]);

    if (filteredProbs.length === 0) continue;

    const subsetBoxes: Box[] = [];

    for (let i = 0; i < numBoxes; i++) {
      if (!mask[i]) continue;
      const offset = i * 4;
      subsetBoxes.push([boxes[offset], boxes[offset + 1], boxes[offset + 2], boxes[offset + 3]]);
    }
    const boxProbs = subsetBoxes.map(
      (box, i) =>
        [box[0], box[1], box[2], box[3], filteredProbs[i]] as [
          number,
          number,
          number,
          number,
          number,
        ]
    );

    const nmsBoxes = hardNMS(boxProbs, iouThreshold, topK);

    for (const b of nmsBoxes) {
      pickedBoxProbs.push(b);
      pickedLabels.push(classIndex);
    }
  }

  if (pickedBoxProbs.length === 0) {
    return { boxes: [], labels: [], probs: [] };
  }

  // Rescale to original image size
  const finalBoxes: Box[] = pickedBoxProbs.map((b) => [
    Math.round(b[0] * width),
    Math.round(b[1] * height),
    Math.round(b[2] * width),
    Math.round(b[3] * height),
  ]);

  const finalProbs = pickedBoxProbs.map((b) => b[4]);
  return { boxes: finalBoxes, labels: pickedLabels, probs: finalProbs };
}
function drawThreeVerticalBoxes(ctx: CanvasRenderingContext2D) {
  const width = ctx.canvas.width;
  const height = ctx.canvas.height;
  const partWidth = width / 3;

  for (let i = 0; i < 3; i++) {
    const x = i * partWidth;
    ctx.strokeStyle = "black";
    ctx.lineWidth = 2;
    ctx.strokeRect(x, 0, partWidth, height);
  }
}
function overlapArea(box1: Box, box2: Box): number {
  const xLeft = Math.max(box1[0], box2[0]);
  const yTop = Math.max(box1[1], box2[1]);
  const xRight = Math.min(box1[2], box2[2]);
  const yBottom = Math.min(box1[3], box2[3]);

  if (xRight < xLeft || yBottom < yTop) {
    return 0; // no overlap
  }

  return (xRight - xLeft) * (yBottom - yTop);
}

function divideCanvasAndOverlap(
  canvasWidth: number,
  canvasHeight: number,
  boundingBoxes: Box[]
): Array<Record<number, number>> {
  const partWidth = canvasWidth / 3;
  const partHeight = canvasHeight / 3;

  // Create the 9 parts as boxes
  const parts: Box[] = [];
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) {
      const x1 = col * partWidth;
      const y1 = row * partHeight;
      const x2 = x1 + partWidth;
      const y2 = y1 + partHeight;
      parts.push([x1, y1, x2, y2]);
    }
  }

  // For each bounding box, calculate overlap with each part
  const results: Array<Record<number, number>> = boundingBoxes.map((box) => {
    const overlapMap: Record<number, number> = {};
    parts.forEach((part, i) => {
      overlapMap[i] = overlapArea(box, part);
    });
    return overlapMap;
  });

  return results;
}

export function getLookAtFromOverlaps(overlaps: Array<Record<number, number>>) {
  let maxPart = -1;
  const defaultPart = 4; // Center part

  if (overlaps.length > 0 && overlaps[0]) {
    const ovlp = overlaps[0];
    let maxArea = -1;

    for (let i = 0; i < 9; i++) {
      const area = ovlp[i] ?? 0;
      if (area > maxArea) {
        maxArea = area;
        maxPart = i;
      }
    }

    if (maxPart === -1) {
      // No positive overlap area found in the first detection
      // console.warn("No overlapping area found in the first detection. Defaulting to center.");
      maxPart = defaultPart;
    }
  } else {
    // No overlaps provided, or overlaps[0] is undefined.
    // console.warn("No overlaps provided. Defaulting to center.");
    maxPart = defaultPart;
  }

  // Ensure maxPart is a valid index (0-8); if not, default to center.
  if (maxPart < 0 || maxPart > 8) {
    maxPart = defaultPart;
  }

  const row = Math.floor(maxPart / 3);
  const col = maxPart % 3;

  const x = (col + 0.5) / 3;
  const y = (row + 0.5) / 3;

  return [x, y];
}

export const YoloWebcam = ({
  componentRef,
  onOverlapChange,
  hiddenVideo=true
}: {
  componentRef?: React.RefObject<HTMLElement>;
  onOverlapChange?: (overlaps: Record<number, number>[]) => any;
  hiddenVideo?:boolean
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const hiddenCanvasRef = useRef<HTMLCanvasElement>(null);
  const [session, setSession] = useState<ort.InferenceSession | null>(null);
  const [modelWidth, setModelWidth] = useState(320);
  const [modelHeight, setModelHeight] = useState(240);
  const [canvaWidth, setCanvaWidth] = useState(640);
  const [canvaHeight, setCanvaHeight] = useState(480);
  const THRESHOLD = 0.5;
  //   const [width, height] = useSize(componentRef)
  //   useEffect(() => {
  //       setCanvaWidth(width);
  //   }, [width]);
  //   useEffect(() => {
  //     setCanvaHeight(height);
  // }, [width]);
  const FRAME_INTERVAL = 100; // ms between inferences

  useEffect(() => {
    const loadModel = async () => {
      const loaded = await ort.InferenceSession.create("/processed_ultraface-320.onnx");
      setSession(loaded);
    };
    loadModel();

    const initWebcam = async () => {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.oncanplay = (()=>{
          videoRef.current&&videoRef.current.play().catch(()=>{});
        })
        
      }
    };
    initWebcam();
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      if (!session || !videoRef.current || !hiddenCanvasRef.current) return;

      const video = videoRef.current;
      const hiddenCanvas = hiddenCanvasRef.current;
      hiddenCanvas.width = modelWidth;
      hiddenCanvas.height = modelHeight;

      const ctx = hiddenCanvas.getContext("2d", {
        willReadFrequently: true,
      });
      if (!ctx) return;
      ctx.scale(-1, 1);
      ctx.drawImage(video, -modelWidth, 0, modelWidth, modelHeight);
      const imageData = ctx.getImageData(0, 0, modelWidth, modelHeight).data;
      const inputTensor = preprocess(imageData, modelWidth, modelHeight);

      session.run({ input: inputTensor }).then((results) => {
        const scores = results.scores.data as Float32Array;
        const _boxes = results.boxes.data as Float32Array;
        const { boxes, labels, probs } = predict(
          canvaWidth,
          canvaHeight,
          scores,
          _boxes,
          THRESHOLD,
          0.5,
          -1
        );
        const detections = postProcess(probs, boxes);
        drawOverlay(detections);
        const overlaps = divideCanvasAndOverlap(canvaWidth, canvaHeight, boxes);
        onOverlapChange && onOverlapChange(overlaps);
        // overlaps.forEach((overlap, idx) => {
        //   console.log(`Bounding Box ${idx + 1}:`);

        //   for (let row = 0; row < 3; row++) {
        //     const rowAreas = [];
        //     for (let col = 0; col < 3; col++) {
        //       const partIndex = row * 3 + col;
        //       rowAreas.push(overlap[partIndex]?.toFixed(2) ?? "0.00");
        //     }
        //     console.log(rowAreas.join(" | "));
        //   }

        //   console.log("");
        // });
      });
    }, FRAME_INTERVAL);

    return () => clearInterval(interval);
  }, [session]);

  function preprocess(imageData: Uint8ClampedArray, width: number, height: number): ort.Tensor {
    const floatData = new Float32Array(3 * width * height);

    for (let i = 0; i < width * height; i++) {
      const r = imageData[i * 4];
      const g = imageData[i * 4 + 1];
      const b = imageData[i * 4 + 2];

      floatData[i] = (r - 127) / 128; // R
      floatData[i + width * height] = (g - 127) / 128; // G
      floatData[i + 2 * width * height] = (b - 127) / 128; // B
    }

    return new ort.Tensor("float32", floatData, [1, 3, height, width]);
  }

  const postProcess = (scores: number[], boxes: Box[]): Detection[] => {
    const detections: Detection[] = [];

    for (let i = 0; i < scores.length; i++) {
      detections.push({ score: scores[i], box: boxes[i] });
    }

    return detections;
  };

  const drawOverlay = (detections: Detection[]) => {
    const canvas = overlayCanvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx || !videoRef.current) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawThreeVerticalBoxes(ctx);
    ctx.strokeStyle = "lime";
    ctx.lineWidth = 2;
    ctx.font = "16px sans-serif";
    ctx.fillStyle = "lime";

    for (const det of detections) {
      const [x1, y1, x2, y2] = det.box;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.fillText(`${(det.score * 100).toFixed(1)}%`, x1, y1);
    }
  };

  return (
    <>
      <video
        ref={videoRef}
        width={canvaWidth}
        height={canvaHeight}
        className="rounded"
        style={{
          display:hiddenVideo?"none":"block",
          position: "fixed",
          bottom: 0,
          left: 0,
          zIndex: 1000,
          transform: "scaleX(-1)",
        }}
      />
      <canvas
        ref={overlayCanvasRef}
        width={canvaWidth}
        height={canvaHeight}
        style={{
          display:hiddenVideo?"none":"block",
          position: "fixed",
          bottom: 0,
          left: 0,
          zIndex: 1000,
        }}
      />
      <canvas ref={hiddenCanvasRef} style={{ display: "none" }} />
    </>
  );
};

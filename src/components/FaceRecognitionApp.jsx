import React, { useRef, useState, useEffect } from "react";
import * as faceapi from "face-api.js";

const FaceRecognitionApp = () => {
  const videoRef = useRef(null);
  const canvasContainerRef = useRef(null);
  const [descriptors, setDescriptors] = useState([]);
  const [name, setName] = useState("");
  const [detecting, setDetecting] = useState(false);
  const [brightnessWarning, setBrightnessWarning] = useState("");
  // Check brightness of the video frame
  const checkBrightness = () => {
    if (!videoRef.current) return;
    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const frame = ctx.getImageData(0, 0, canvas.width, canvas.height);
    let total = 0;
    for (let i = 0; i < frame.data.length; i += 4) {
      // Average of R, G, B
      total += (frame.data[i] + frame.data[i + 1] + frame.data[i + 2]) / 3;
    }
    const avg = total / (frame.data.length / 4);
    // Threshold: below 60 is considered too dark
    if (avg < 60) {
      setBrightnessWarning(
        "Warning: Camera image is too dark. Please increase brightness."
      );
    } else {
      setBrightnessWarning("");
    }
  };

  // Load models and start webcam
  useEffect(() => {
    const loadModels = async () => {
      await faceapi.nets.tinyFaceDetector.loadFromUri("/models");
      await faceapi.nets.faceLandmark68Net.loadFromUri("/models");
      await faceapi.nets.faceRecognitionNet.loadFromUri("/models");
      startVideo();
    };
    loadModels();

    // Load descriptors from localStorage
    const saved = localStorage.getItem("face_descriptors");
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        // Recreate LabeledFaceDescriptors
        const loaded = parsed.map(
          (item) =>
            new faceapi.LabeledFaceDescriptors(
              item.label,
              item.descriptors.map((desc) => new Float32Array(desc))
            )
        );
        setDescriptors(loaded);
      } catch (e) {
        console.error("Failed to load face descriptors from storage", e);
      }
    }
  }, []);

  // Start webcam stream
  const startVideo = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: {} });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      console.error("Error accessing webcam:", err);
      alert("Please allow webcam access");
    }
  };

  // Register face with entered name
  const handleRegister = async () => {
    if (!name) {
      alert("Please enter a name before registering!");
      return;
    }

    const detection = await faceapi
      .detectSingleFace(videoRef.current, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceDescriptor();

    if (!detection) {
      alert("No face detected. Please try again.");
      return;
    }

    const newDescriptor = new faceapi.LabeledFaceDescriptors(name, [
      detection.descriptor,
    ]);
    setDescriptors((prev) => {
      const updated = [...prev, newDescriptor];
      // Save to localStorage as JSON
      const toSave = updated.map((desc) => ({
        label: desc.label,
        descriptors: desc.descriptors.map((d) => Array.from(d)),
      }));
      localStorage.setItem("face_descriptors", JSON.stringify(toSave));
      return updated;
    });
    alert(`${name} registered successfully!`);
    setName("");
  };

  // Start detection and recognition
  const handleDetect = async () => {
    if (descriptors.length === 0) {
      alert("Please register at least one face first!");
      return;
    }

    setDetecting(true);
    const faceMatcher = new faceapi.FaceMatcher(descriptors, 0.6);

    const canvas = faceapi.createCanvasFromMedia(videoRef.current);
    const container = canvasContainerRef.current;
    container.innerHTML = "";
    container.appendChild(canvas);

    const displaySize = {
      width: videoRef.current.videoWidth,
      height: videoRef.current.videoHeight,
    };
    faceapi.matchDimensions(canvas, displaySize);

    const detectInterval = setInterval(async () => {
      checkBrightness();
      const detections = await faceapi
        .detectAllFaces(videoRef.current, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptors();

      const resized = faceapi.resizeResults(detections, displaySize);
      canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);

      resized.forEach((detection) => {
        const match = faceMatcher.findBestMatch(detection.descriptor);
        const box = detection.detection.box;
        // If match label starts with 'unknown', show 'Unknown' only
        let label = match.label === "unknown" ? "Unknown" : match.toString();
        const drawBox = new faceapi.draw.DrawBox(box, { label });
        drawBox.draw(canvas);
      });
    }, 300);

    // Stop after 30s
    setTimeout(() => {
      clearInterval(detectInterval);
      setDetecting(false);
    }, 30000);
  };

  return (
    <div style={{ textAlign: "center", marginTop: 20 }}>
      <h2>Face Recognition App</h2>
      {brightnessWarning && (
        <div style={{ color: "orange", marginBottom: 10 }}>
          {brightnessWarning}
        </div>
      )}
      <div style={{ position: "relative", display: "inline-block" }}>
        <video
          ref={videoRef}
          width="640"
          height="480"
          autoPlay
          muted
          style={{ borderRadius: 10 }}
        />
        <div
          ref={canvasContainerRef}
          style={{ position: "absolute", top: 0, left: 0 }}
        />
      </div>

      <div style={{ marginTop: 20 }}>
        <input
          type="text"
          placeholder="Enter your name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          disabled={detecting}
        />
        <button onClick={handleRegister} disabled={detecting}>
          Register
        </button>
        <button onClick={handleDetect} disabled={detecting}>
          Detect
        </button>
      </div>
    </div>
  );
};

export default FaceRecognitionApp;

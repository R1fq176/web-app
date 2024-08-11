// Import dependencies
import { useRef, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import Webcam from "react-webcam";
// import "./App.css";
// 2. TODO - Import drawing utility here
// e.g. import { drawRect } from "./utilities";
import { drawRect } from "@/lib/draw";

function CameraDetection(): JSX.Element {
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const detect = async (net: tf.GraphModel): Promise<void> => {
    // Check data is available
    if (typeof webcamRef.current !== "undefined" && webcamRef.current !== null && webcamRef.current.video !== null && webcamRef.current.video.readyState === 4) {
      // Get Video Properties
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      // Set video width
      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;

      // Set canvas height and width
      if (canvasRef.current) {
        canvasRef.current.width = videoWidth;
        canvasRef.current.height = videoHeight;
      }

      // 4. TODO - Make Detections
      const img = tf.browser.fromPixels(video);
      const resized = tf.image.resizeBilinear(img, [640, 480]);
      const casted = resized.cast("int32");
      const expanded = casted.expandDims(0);
      const obj = (await net.executeAsync(expanded)) as tf.Tensor<tf.Rank>[];
      console.log(obj);

      const boxes = (await obj[1].array()) as number[][][];
      const classes = (await obj[2].array()) as number[][];
      const scores = (await obj[4].array()) as number[][];

      const ctx = canvasRef.current?.getContext("2d");

      if (ctx) {
        requestAnimationFrame(() => {
          drawRect(boxes[0], classes[0], scores[0], 0.8, videoWidth, videoHeight, ctx);
        });
      }

      tf.dispose(img);
      tf.dispose(resized);
      tf.dispose(casted);
      tf.dispose(expanded);
      tf.dispose(obj);
    }
  };

  useEffect(() => {
    // Main function
    const runCoco = async (): Promise<void> => {
      // 3. TODO - Load network
      // e.g. const net = await cocossd.load();
      // https://tensorflowjsrealtimemodel.s3.au-syd.cloud-object-storage.appdomain.cloud/model.json
      const net = await tf.loadGraphModel("https://tensorflowjsrealtimemodel.s3.au-syd.cloud-object-storage.appdomain.cloud/model.json");

      //  Loop and detect hands
      setInterval(() => {
        detect(net);
      }, 16.7);
    };
    runCoco();
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        {/* <Webcam
          ref={webcamRef}
          muted={true}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zIndex: 9,
            width: 640,
            height: 480,
          }}
        /> */}
        <Webcam
          audio={false}
          muted={true}
          ref={webcamRef}
          screenshotFormat="image/png"
          className="max-sm:w-[98%] max-sm:max-h-[300px] sm:h-[300px] md:h-[400px] 2xl:h-[500px] rounded-2xl overflow-hidden"
        />
        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zIndex: 8,
            width: 640,
            height: 480,
          }}
        />
      </header>
    </div>
  );
}

export default CameraDetection;

import React from "react";
import ReactDOM from "react-dom/client";
import "./index.css";
import Navbar from "./components/navbar.tsx";
import AppRouter from "./Router.tsx";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    {/* Global layout */}
    <div className="flex w-screen h-screen overflow-x-hidden relative">
      <Navbar />
      {/* Main page */}
      <AppRouter />
      {/* Footer */}
    </div>
  </React.StrictMode>
);

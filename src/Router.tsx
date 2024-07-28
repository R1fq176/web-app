// src/Router.jsx
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import NotFound from "./pages/404";
import KosakataPage from "./pages/Kosakata";
import KataPage from "./pages/search/KataPage";
import KalimatPage from "./pages/search/KalimatPage";
import AboutPage from "./pages/About";

function AppRouter() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/kosakata" element={<KosakataPage />} />
        <Route path="/tentang" element={<AboutPage />} />
        <Route path="/search/kata" element={<KataPage />} />
        <Route path="/search/kalimat" element={<KalimatPage />} />

        <Route path="*" element={<NotFound />} />
      </Routes>
    </Router>
  );
}

export default AppRouter;

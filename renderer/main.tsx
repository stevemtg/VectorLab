import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import Page from "../app/page";
import "../app/globals.css";

const container = document.getElementById("root");
if (!container) throw new Error("Vector Lab renderer is missing its #root container.");
createRoot(container).render(<StrictMode><Page /></StrictMode>);

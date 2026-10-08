// Sandboxed preloads use CommonJS even though package.json is ESM.
/* eslint-disable @typescript-eslint/no-require-imports */
"use strict";
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("vectorLab", {
  desktop: true,
  platform: process.platform,
  getServiceConnection: () => ipcRenderer.invoke("vectorlab:get-service-connection"),
  getServiceStatus: () => ipcRenderer.invoke("vectorlab:get-service-status"),
  onServiceStatus: callback => {
    if (typeof callback !== "function") throw new TypeError("Expected a callback.");
    const listener = (_event, status) => callback(status); // Never expose the IPC event.
    ipcRenderer.on("vectorlab:service-status", listener);
    return () => ipcRenderer.removeListener("vectorlab:service-status", listener);
  },
  showNotification: (title, body) => ipcRenderer.invoke("vectorlab:show-notification", { title, body }),
  importVectors: () => ipcRenderer.invoke("vectorlab:import-vectors"),
});

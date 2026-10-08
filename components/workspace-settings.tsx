"use client";

import { RotateCcw, SlidersHorizontal } from "lucide-react";
import { Dialog, DialogClose, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Switch } from "@/components/ui/switch";

export const DEFAULT_WORKSPACE_PREFERENCES = {
  autoScroll: true,
  showSuggestions: true,
  showTelemetry: true,
  completionNotifications: false,
};
export type WorkspacePreferences = typeof DEFAULT_WORKSPACE_PREFERENCES;
export const WORKSPACE_PREFERENCES_KEY = "vector-lab.workspace-preferences";

export function readWorkspacePreferences(): WorkspacePreferences {
  try {
    const saved = JSON.parse(localStorage.getItem(WORKSPACE_PREFERENCES_KEY) ?? "null") as Partial<WorkspacePreferences> | null;
    return {
      autoScroll: typeof saved?.autoScroll === "boolean" ? saved.autoScroll : true,
      showSuggestions: typeof saved?.showSuggestions === "boolean" ? saved.showSuggestions : true,
      showTelemetry: typeof saved?.showTelemetry === "boolean" ? saved.showTelemetry : true,
      completionNotifications: typeof saved?.completionNotifications === "boolean" ? saved.completionNotifications : false,
    };
  } catch {
    return DEFAULT_WORKSPACE_PREFERENCES;
  }
}

const options: { key: keyof WorkspacePreferences; label: string; description: string }[] = [
  { key: "autoScroll", label: "Auto-scroll responses", description: "Follow new output while you are at the bottom of the chat." },
  { key: "showSuggestions", label: "Suggested prompts", description: "Show starter prompts in an empty chat." },
  { key: "showTelemetry", label: "Telemetry panel", description: "Show internal measurements and the model event stream." },
];

export function WorkspaceSettings({ preferences, onChange }: { preferences: WorkspacePreferences; onChange: (preferences: WorkspacePreferences) => void }) {
  return <Dialog>
    <DialogTrigger asChild>
      <button type="button" className="workspace-symbol" aria-label="Workspace settings" title="Workspace settings">
        <SlidersHorizontal size={24} aria-hidden="true"/>
      </button>
    </DialogTrigger>
    <DialogContent className="workspace-settings">
      <DialogHeader>
        <DialogTitle className="workspace-settings-title"><SlidersHorizontal size={19} aria-hidden="true"/>Workspace settings</DialogTitle>
        <DialogDescription>Adjust your workspace. Changes apply immediately and save in this browser.</DialogDescription>
      </DialogHeader>
      <div className="workspace-settings-options">
        {options.map(option => <div className="workspace-setting" key={option.key}>
          <div>
            <label htmlFor={`workspace-${option.key}`}>{option.label}</label>
            <p id={`workspace-${option.key}-description`}>{option.description}</p>
          </div>
          <Switch id={`workspace-${option.key}`} aria-describedby={`workspace-${option.key}-description`} checked={preferences[option.key]} onCheckedChange={checked => onChange({ ...preferences, [option.key]: checked })}/>
        </div>)}
      </div>
      {typeof window !== "undefined" && window.vectorLab && <div className="workspace-setting">
        <div><label htmlFor="workspace-completion-notifications">Completion notifications</label><p>Notify when a response finishes while Vector Lab is in the background.</p></div>
        <Switch id="workspace-completion-notifications" checked={preferences.completionNotifications} onCheckedChange={checked => onChange({ ...preferences, completionNotifications: checked })}/>
      </div>}
      <DialogFooter className="workspace-settings-footer">
        <button type="button" className="text-button" onClick={() => onChange(DEFAULT_WORKSPACE_PREFERENCES)}><RotateCcw size={14} aria-hidden="true"/>Reset defaults</button>
        <DialogClose asChild><button type="button" className="primary-button">Done</button></DialogClose>
      </DialogFooter>
    </DialogContent>
  </Dialog>;
}

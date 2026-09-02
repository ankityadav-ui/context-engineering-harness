import { useState, useEffect } from "react";
import { Check, ChevronDown, Loader2 } from "lucide-react";
import { api } from "../api/client";

function Settings() {
  const [providers, setProviders] = useState({});
  const [currentProvider, setCurrentProvider] = useState("");
  const [currentModel, setCurrentModel] = useState("");
  const [selectedProvider, setSelectedProvider] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState(null);

  useEffect(() => {
    async function load() {
      try {
        const [provData, settingsData] = await Promise.all([
          api.get("/settings/llm/providers"),
          api.get("/settings/llm"),
        ]);
        setProviders(provData.providers || {});
        setCurrentProvider(settingsData.provider || "");
        setCurrentModel(settingsData.model || "");
        setSelectedProvider(settingsData.provider || "");
        setSelectedModel(settingsData.model || "");
      } catch (err) {
        console.error("Failed to load LLM settings:", err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const handleSave = async () => {
    setSaving(true);
    setMessage(null);
    try {
      const data = await api.put("/settings/llm", {
        provider: selectedProvider,
        model: selectedModel,
      });
      setCurrentProvider(data.provider);
      setCurrentModel(data.model);
      setMessage({ type: "success", text: data.message });
      // Notify Runtime panel and other listeners that settings changed
      window.dispatchEvent(new Event("llm-settings-changed"));
    } catch (err) {
      setMessage({ type: "error", text: err.message || "Failed to save settings" });
    } finally {
      setSaving(false);
    }
  };

  const hasChanges =
    selectedProvider !== currentProvider ||
    selectedModel !== currentModel;

  const models =
    selectedProvider && providers[selectedProvider]
      ? providers[selectedProvider].models
      : [];

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="animate-spin text-zinc-500" size={24} />
      </div>
    );
  }

  return (
    <div className="p-6 max-w-2xl">
      <h1 className="text-3xl font-semibold">Settings</h1>
      <p className="mt-2 text-zinc-400">
        Configure your Harness LLM provider and model.
      </p>

      {/* Provider Selection */}
      <div className="mt-8">
        <h3 className="text-sm font-medium text-zinc-300 mb-3">
          LLM Provider
        </h3>
        <div className="grid grid-cols-2 gap-3">
          {Object.entries(providers).map(([key, info]) => (
            <button
              key={key}
              onClick={() => {
                setSelectedProvider(key);
                setSelectedModel(info.default_model);
                setMessage(null);
              }}
              className={`rounded-lg border p-4 text-left transition ${
                selectedProvider === key
                  ? "border-white bg-white/10 text-white"
                  : "border-zinc-800 bg-[#1a1a1a] text-zinc-400 hover:border-zinc-600"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">
                  {info.name}
                </span>
                {selectedProvider === key && (
                  <Check size={16} className="text-white" />
                )}
              </div>
              <p className="mt-1 text-xs text-zinc-500">
                {info.models.length} models available
              </p>
            </button>
          ))}
        </div>
      </div>

      {/* Model Selection */}
      {selectedProvider && (
        <div className="mt-8">
          <h3 className="text-sm font-medium text-zinc-300 mb-3">
            Model
          </h3>
          <div className="relative">
            <select
              value={selectedModel}
              onChange={(e) => {
                setSelectedModel(e.target.value);
                setMessage(null);
              }}
              className="w-full appearance-none rounded-lg border border-zinc-800 bg-[#1a1a1a] px-4 py-3 text-sm text-white focus:border-zinc-600 focus:outline-none"
            >
              {models.map((model) => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
            <ChevronDown
              size={16}
              className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-zinc-500"
            />
          </div>
        </div>
      )}

      {/* Save Button */}
      <div className="mt-8 flex items-center gap-4">
        <button
          onClick={handleSave}
          disabled={!hasChanges || saving}
          className={`rounded-lg px-6 py-2.5 text-sm font-medium transition ${
            hasChanges && !saving
              ? "bg-white text-black hover:bg-zinc-200"
              : "bg-zinc-800 text-zinc-500 cursor-not-allowed"
          }`}
        >
          {saving ? (
            <span className="flex items-center gap-2">
              <Loader2 size={14} className="animate-spin" />
              Saving...
            </span>
          ) : (
            "Save Changes"
          )}
        </button>
        {message && (
          <span
            className={`text-sm ${
              message.type === "success"
                ? "text-green-400"
                : "text-red-400"
            }`}
          >
            {message.text}
          </span>
        )}
      </div>

      {/* Current Config */}
      <div className="mt-12 rounded-lg border border-zinc-800 bg-[#1a1a1a] p-5">
        <h3 className="text-sm font-medium text-zinc-300 mb-3">
          Current Configuration
        </h3>
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-zinc-500">Provider</span>
            <span className="text-white font-medium">
              {currentProvider || "—"}
            </span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-zinc-500">Model</span>
            <span className="text-white font-medium">
              {currentModel || "—"}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Settings;
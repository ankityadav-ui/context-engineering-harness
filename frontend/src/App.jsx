import {
  Plus,
  MessageSquare,
  FolderOpen,
  FileText,
  Network,
  Settings as SettingsIcon,
  Sparkles,
  ArrowUp,
  Cpu,
  SlidersHorizontal,
  Brain,
  Users,
} from "lucide-react";

import {
  BrowserRouter,
  Routes,
  Route,
  Link,
} from "react-router-dom";

import Playground from "./pages/Playground";
import Chats from "./pages/Chats";
import Cases from "./pages/Cases";
import Documents from "./pages/Documents";
import KnowledgeGraph from "./pages/KnowledgeGraph";
import Settings from "./pages/Settings";

function App() {
  return (
    <div className="flex h-screen bg-[#111111] text-white">

      {/* ================= SIDEBAR ================= */}
      <aside className="flex w-64 flex-col border-r border-zinc-800 bg-[#151515] p-4">

        {/* Logo */}
        <div className="mb-8 flex items-center gap-2 px-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-white text-black">
            <Sparkles size={18} />
          </div>

          <span className="text-lg font-semibold">
            HARNESS
          </span>
        </div>

        {/* New Chat */}
        <button className="mb-6 flex items-center justify-center gap-2 rounded-lg bg-white px-4 py-2.5 font-medium text-black transition hover:bg-zinc-200">
          <Plus size={18} />
          New Chat
        </button>

        {/* Navigation */}
        <nav className="flex flex-col gap-1">

          <button className="flex items-center gap-3 rounded-lg bg-zinc-800 px-3 py-2.5 text-sm text-white">
            <MessageSquare size={18} />
            Playground
          </button>

          <button className="flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm text-zinc-400 transition hover:bg-zinc-800 hover:text-white">
            <MessageSquare size={18} />
            Chats
          </button>

          <button className="flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm text-zinc-400 transition hover:bg-zinc-800 hover:text-white">
            <FolderOpen size={18} />
            Cases
          </button>

          <button className="flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm text-zinc-400 transition hover:bg-zinc-800 hover:text-white">
            <FileText size={18} />
            Documents
          </button>

          <button className="flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm text-zinc-400 transition hover:bg-zinc-800 hover:text-white">
            <Network size={18} />
            Knowledge Graph
          </button>

        </nav>

        {/* Settings */}
        <div className="mt-auto border-t border-zinc-800 pt-4">

          <button className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm text-zinc-400 transition hover:bg-zinc-800 hover:text-white">
            <Settings size={18} />
            Settings
          </button>

        </div>

      </aside>


      {/* ================= MAIN ================= */}
      <main className="flex min-w-0 flex-1 flex-col">

        {/* Header */}
        <header className="flex h-16 items-center justify-between border-b border-zinc-800 px-6">

          <h2 className="text-lg font-medium">
            Playground
          </h2>

          <div className="text-sm text-zinc-500">
            Context Engineering Harness
          </div>

        </header>


        {/* ================= CHAT AREA ================= */}
        <section className="flex flex-1 flex-col">

          {/* Empty State */}
          <div className="flex flex-1 items-center justify-center">

            <div className="max-w-xl px-6 text-center">

              <div className="mb-6 flex justify-center">

                <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-zinc-700 bg-zinc-900">

                  <Sparkles size={26} />

                </div>

              </div>

              <h2 className="mb-3 text-4xl font-semibold">
                How can I help?
              </h2>

              <p className="text-zinc-400">
                Ask questions, analyze documents, or work with
                your configured AI system.
              </p>

            </div>

          </div>


          {/* Chat Input */}
          <div className="px-6 pb-6">

            <div className="mx-auto max-w-3xl">

              <div className="flex items-end rounded-2xl border border-zinc-700 bg-[#1a1a1a] p-3 shadow-lg">

                <textarea
                  placeholder="Ask anything..."
                  rows="1"
                  className="flex-1 resize-none bg-transparent px-3 py-2 text-sm text-white outline-none placeholder:text-zinc-500"
                />

                <button className="flex h-9 w-9 items-center justify-center rounded-lg bg-white text-black transition hover:bg-zinc-200">
                  <ArrowUp size={18} />
                </button>

              </div>

              <p className="mt-2 text-center text-xs text-zinc-600">
                Harness can make mistakes. Verify important information.
              </p>

            </div>

          </div>

        </section>

      </main>



      {/* ================= RUNTIME PANEL ================= */}
      <aside className="hidden w-72 flex-col border-l border-zinc-800 bg-[#151515] lg:flex">

        {/* Runtime Header */}
        <div className="flex h-16 items-center border-b border-zinc-800 px-5">

          <div className="flex items-center gap-2">

            <SlidersHorizontal size={18} />

            <span className="font-medium">
              Runtime
            </span>

          </div>

        </div>


        {/* Runtime Content */}
        <div className="flex-1 overflow-y-auto p-4">


          {/* LLM */}
          <div className="mb-6">

            <div className="mb-2 flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-zinc-500">

              <Cpu size={14} />

              LLM

            </div>


            <div className="rounded-lg border border-zinc-800 bg-[#1a1a1a] p-3">

              <p className="text-sm font-medium">
                Provider
              </p>

              <p className="mt-1 text-xs text-zinc-500">
                Select provider
              </p>

            </div>


            <div className="mt-2 rounded-lg border border-zinc-800 bg-[#1a1a1a] p-3">

              <p className="text-sm font-medium">
                Model
              </p>

              <p className="mt-1 text-xs text-zinc-500">
                Select model
              </p>

            </div>

          </div>


          {/* System Prompt */}
          <div className="mb-6">

            <div className="mb-2 flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-zinc-500">

              <Brain size={14} />

              System Prompt

            </div>

            <button className="w-full rounded-lg border border-zinc-800 bg-[#1a1a1a] p-3 text-left transition hover:border-zinc-600">

              <p className="text-sm font-medium">
                Configure prompt
              </p>

              <p className="mt-1 text-xs text-zinc-500">
                Define AI behavior
              </p>

            </button>

          </div>


          {/* Skills */}
          <div className="mb-6">

            <div className="mb-2 text-xs font-medium uppercase tracking-wide text-zinc-500">
              Skills
            </div>

            <button className="w-full rounded-lg border border-zinc-800 bg-[#1a1a1a] p-3 text-left transition hover:border-zinc-600">

              <p className="text-sm font-medium">
                skills.md
              </p>

              <p className="mt-1 text-xs text-zinc-500">
                No skills configured
              </p>

            </button>

          </div>


          {/* Config */}
          <div className="mb-6">

            <div className="mb-2 text-xs font-medium uppercase tracking-wide text-zinc-500">
              Configuration
            </div>

            <button className="w-full rounded-lg border border-zinc-800 bg-[#1a1a1a] p-3 text-left transition hover:border-zinc-600">

              <p className="text-sm font-medium">
                config.yaml
              </p>

              <p className="mt-1 text-xs text-zinc-500">
                Runtime configuration
              </p>

            </button>

          </div>


          {/* Context */}
          <div className="mb-6">

            <div className="mb-2 flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-zinc-500">

              <Brain size={14} />

              Context

            </div>

            <div className="rounded-lg border border-zinc-800 bg-[#1a1a1a] p-3">

              <div className="flex items-center justify-between">

                <span className="text-sm">
                  Memory
                </span>

                <span className="text-xs text-green-400">
                  Enabled
                </span>

              </div>

            </div>

          </div>


          {/* Agents */}
          <div>

            <div className="mb-2 flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-zinc-500">

              <Users size={14} />

              Agents

            </div>

            <div className="rounded-lg border border-zinc-800 bg-[#1a1a1a] p-3">

              <div className="flex items-center justify-between">

                <span className="text-sm">
                  Max agents
                </span>

                <span className="text-sm font-medium">
                  3
                </span>

              </div>

              <p className="mt-1 text-xs text-zinc-500">
                Spawn only when required
              </p>

            </div>

          </div>


        </div>

      </aside>

    

    </div>
  );
}

export default App;
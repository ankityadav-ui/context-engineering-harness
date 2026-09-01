import {
  Plus,
  MessageSquare,
  FolderOpen,
  FileText,
  Network,
  Settings as SettingsIcon,
  Sparkles,
  SlidersHorizontal,
  Brain,
  Users,
  Cpu,
  BarChart3,
} from "lucide-react";

import {
  BrowserRouter,
  Routes,
  Route,
  Link,
  useLocation,
} from "react-router-dom";

import Playground from "./pages/Playground";
import Chats from "./pages/Chats";
import Cases from "./pages/Cases";
import Documents from "./pages/Documents";
import KnowledgeGraph from "./pages/KnowledgeGraph";
import Settings from "./pages/Settings";
import RAGEvaluation from "./pages/RAGEvaluation";
import Memory from "./pages/Memory";


function Layout() {
  const location = useLocation();

  const pageTitles = {
    "/": "Playground",
    "/playground": "Playground",
    "/chats": "Chats",
    "/cases": "Cases",
    "/documents": "Documents",
    "/knowledge-graph": "Knowledge Graph",
    "/rag-evaluation": "RAG Evaluation",
    "/memory": "Memory",
    "/settings": "Settings",
  };

  const currentTitle =
    pageTitles[location.pathname] || "Playground";


  const navItems = [
    {
      name: "Playground",
      path: "/playground",
      icon: MessageSquare,
    },
    {
      name: "Chats",
      path: "/chats",
      icon: MessageSquare,
    },
    {
      name: "Cases",
      path: "/cases",
      icon: FolderOpen,
    },
    {
      name: "Documents",
      path: "/documents",
      icon: FileText,
    },
    {
      name: "Knowledge Graph",
      path: "/knowledge-graph",
      icon: Network,
    },
    {
      name: "RAG Evaluation",
      path: "/rag-evaluation",
      icon: BarChart3,
    },
    {
      name: "Memory",
      path: "/memory",
      icon: Brain,
    },
  ];


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

        <Link
          to="/chats"
          className="mb-6 flex items-center justify-center gap-2 rounded-lg bg-white px-4 py-2.5 font-medium text-black transition hover:bg-zinc-200"
        >
          <Plus size={18} />
          New Chat
        </Link>


        {/* Navigation */}

        <nav className="flex flex-col gap-1">

          {navItems.map((item) => {

            const Icon = item.icon;

            const isActive =
              location.pathname === item.path ||
              (
                item.path === "/playground" &&
                location.pathname === "/"
              );

            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition ${
                  isActive
                    ? "bg-zinc-800 text-white"
                    : "text-zinc-400 hover:bg-zinc-800 hover:text-white"
                }`}
              >
                <Icon size={18} />

                {item.name}

              </Link>
            );
          })}

        </nav>


        {/* Settings */}

        <div className="mt-auto border-t border-zinc-800 pt-4">

          <Link
            to="/settings"
            className={`flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition ${
              location.pathname === "/settings"
                ? "bg-zinc-800 text-white"
                : "text-zinc-400 hover:bg-zinc-800 hover:text-white"
            }`}
          >
            <SettingsIcon size={18} />

            Settings

          </Link>

        </div>

      </aside>


      {/* ================= MAIN ================= */}

      <main className="flex min-w-0 flex-1 flex-col">

        {/* Header */}

        <header className="flex h-16 items-center justify-between border-b border-zinc-800 px-6">

          <h2 className="text-lg font-medium">
            {currentTitle}
          </h2>

          <div className="text-sm text-zinc-500">
            Context Engineering Harness
          </div>

        </header>


        {/* Page Content */}

        <div className="flex min-h-0 flex-1">

          <div className="min-w-0 flex-1 overflow-hidden">

            <Routes>

              <Route
                path="/"
                element={<Playground />}
              />

              <Route
                path="/playground"
                element={<Playground />}
              />

              <Route
                path="/chats"
                element={<Chats />}
              />

              <Route
                path="/cases"
                element={<Cases />}
              />

              <Route
                path="/documents"
                element={<Documents />}
              />

              <Route
                path="/knowledge-graph"
                element={<KnowledgeGraph />}
              />

              <Route
                path="/rag-evaluation"
                element={<RAGEvaluation />}
              />

              <Route
                path="/memory"
                element={<Memory />}
              />

              <Route
                path="/settings"
                element={<Settings />}
              />

            </Routes>

          </div>


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
                    Gemini
                  </p>

                </div>


                <div className="mt-2 rounded-lg border border-zinc-800 bg-[#1a1a1a] p-3">

                  <p className="text-sm font-medium">
                    Model
                  </p>

                  <p className="mt-1 text-xs text-zinc-500">
                    Gemini 3.6 Flash
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


              {/* Configuration */}

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

      </main>

    </div>
  );
}


function App() {
  return (
    <BrowserRouter>
      <Layout />
    </BrowserRouter>
  );
}


export default App;